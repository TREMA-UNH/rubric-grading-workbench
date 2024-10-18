import itertools
from pathlib import Path
from typing import Tuple, List, Any, Dict, Optional
import hashlib
import re
import itertools
import re

from collections import defaultdict
from pydantic.v1 import BaseModel



import trec_car.read_data as trec_car
from .data_model import *
from . import data_model
from .pydantic_helper import pydantic_dump


def get_md5_hash(input_string: str) -> str:
    # Convert the string to bytes
    input_bytes = input_string.encode('utf-8')

    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Update the hash object with the bytes
    md5_hash.update(input_bytes)

    # Get the hexadecimal digest of the hash
    hex_digest = md5_hash.hexdigest()

    return hex_digest


# -------- Parse Davinci JSONL format ----------------

class DavinciResponse(BaseModel):
    datatime:str
    response:str
    gptmodel:str
    prompt:str
    benchmark:str
    queryId:str
    queryStr:str

    sectionQueryStr:Optional[str]
    pageOrSection:Optional[str]

    def isSection(self)->bool:
        if self.pageOrSection == "section":
            return True
        return False


def parse_davinci_response_line(line:str) -> DavinciResponse:
    # Parse the JSON content of the line
    return DavinciResponse.parse_raw(line)


# Path to the benchmarkY3test-qrels-with-text.jsonl.gz file
def parse_davinci_response_file(file_path:Path) -> List[DavinciResponse]:
    '''Load JSONL file with car-specific responses from GPT davinci-003 model'''
    with open(file_path, 'rt', encoding='utf-8') as file:
        return list([parse_davinci_response_line(line) for line in file])
    
    return []


# --------- load TQA information to get queryID info ---------------

def parse_davinci_into_dict(page_file_path:Optional[Path], section_file_path:Optional[Path]) -> Dict[str, List[DavinciResponse]]:

    davinci_by_query_id = defaultdict(list)    

    if page_file_path is not None:
        davinci_pages = parse_davinci_response_file(file_path=page_file_path)
        for resp in davinci_pages:
            davinci_by_query_id[resp.queryId].append(resp)

    if section_file_path is not None:
        davinci_sections = parse_davinci_response_file(file_path=section_file_path)
        for resp in davinci_sections:
            davinci_by_query_id[resp.queryId].append(resp)


    if len(davinci_by_query_id) == 0:
        raise RuntimeError(f'No davinci responses collected from input files. Page: {page_file_path}, Section: {section_file_path}')


    return davinci_by_query_id



def davinci_response_to_full_paragraph(query_id:str, query_facet_id:Optional[str], davinci_response:DavinciResponse)->List[FullParagraphData]:
    result:List[FullParagraphData] = list()
    page_text:str = davinci_response.response

    # Preprocessing
    # 1. replace  markdown headings like "==History==" with empty strings
    # 2. strip leading/trailing whitespace
    # 3. break text at `\n\n`
    # 4. filter out empty strings
    page_para_texts = [txt.strip() for txt in re.sub(r"==.*==\n", '', page_text).strip().split("\n\n") if len(txt)>0]

    # turn each chunk into its own paragraph
    for index, page_para_text in enumerate(page_para_texts):
        # print(f'{query_id}/{query_facet_id} {len(page_para_text)} {page_para_text}')
        # print(f'{len(page_para_text)} {page_para_text}')
        
        page_para_id = f'davinci3:{get_md5_hash(page_para_text)}'
        rank = index+1
        ranking_query_id = query_id if query_facet_id is None else f'{query_id}/{query_facet_id}'
        pdata = ParagraphData(judgments=[]
                            , rankings=[ParagraphRankingEntry(method=davinci_response.gptmodel
                                                            , paragraphId=page_para_id
                                                            , queryId=ranking_query_id
                                                            , rank=rank
                                                            , score=1.0/(1.0*(rank)))
                                                    ])
        fp = FullParagraphData(paragraph_id=page_para_id
                                        , text=page_para_text
                                        , paragraph=None
                                        , paragraph_data=pdata
                                        , exam_grades=None
                                        , grades = None
                                        )
        result.append(fp)
                           
    return result


def parse_davinci_as_query_with_full_paragraph_list(section_davinci_path:Optional[Path], page_davinci_path:Optional[Path], car_outlines_path:Path, max_queries:Optional[int]=None) -> List[QueryWithFullParagraphList]:
    result:List[QueryWithFullParagraphList]=list()
    davinci_by_query_id = parse_davinci_into_dict(section_file_path=section_davinci_path, page_file_path=page_davinci_path)


    page:trec_car.Page
    for page in itertools.islice(trec_car.iter_outlines(open(car_outlines_path, 'rb')), max_queries):
        query_id = page.page_id

        paragraphs:List[Any] = list()

        # page text
        davinci_pages = davinci_by_query_id.get(query_id, None) 
        if(davinci_pages is not None):
            for davinci_section in davinci_pages: # there should be exactly 1
                para = davinci_response_to_full_paragraph(query_id=query_id, davinci_response=davinci_section, query_facet_id=None)
                paragraphs.extend(para)
        #~ page text

        section:trec_car.Section
        for section in page.child_sections:
            query_facet_id = section.headingId
            print(query_id, query_facet_id) 

            # section text
            davinci_sections = davinci_by_query_id.get(query_facet_id, None)  
            if(davinci_sections is not None):
                for davinci_section in davinci_sections: # there should be 1, but we can also handle more
                    para = davinci_response_to_full_paragraph(query_id=query_id, davinci_response=davinci_section, query_facet_id=query_facet_id)
                    paragraphs.extend(para)
            
            #~ section text

        qp = QueryWithFullParagraphList(queryId=query_id, paragraphs=paragraphs)

        result.append(qp)

    return result


# ---------------------------------



def simple_davinci_to_full_paragraph_list(page_davinci_path:Optional[Path], max_queries:Optional[int]=None, run_file:Optional[Path]=None) -> List[QueryWithFullParagraphList]:
    result:List[QueryWithFullParagraphList]=list()
    run_entries:List[str] = list()


    davinci_by_query_id = parse_davinci_into_dict(section_file_path=None, page_file_path=page_davinci_path)

    for query_id, davinci_pages in itertools.islice(davinci_by_query_id.items(), max_queries):

        paragraphs:List[Any] = list()

        if(davinci_pages is not None):
            for davinci_section in davinci_pages: # there should be exactly 1
                para = davinci_response_to_full_paragraph(query_id=query_id, davinci_response=davinci_section, query_facet_id=None)
                paragraphs.extend(para)
                run_entries.append("\n".join( [  
                                            "\t".join([query_id, "Q0", p.paragraph_id, f"{p.paragraph_data.rankings[0].rank}", f"{p.paragraph_data.rankings[0].score}", p.paragraph_data.rankings[0].method  ])
                                            for p in para])
                                      )
        #~ page text

        qp = QueryWithFullParagraphList(queryId=query_id, paragraphs=paragraphs)

        result.append(qp)
    if run_file is not None:
        with open(run_file, 'wt', encoding="utf-8") as runfile:
            runfile.write("\n".join(run_entries))    
            runfile.flush()
            runfile.close()

    return result



# --------------------------------
    
def main_test():
    page_davinci_path = "./v24-20-lucene-page--text-davinci-003-benchmarkY3test.jsonl"
    section_davinci_path = "./v24-20-lucene-section--text-davinci-003-benchmarkY3test.jsonl"
    car_outlines_path = "./benchmarkY3test.public/benchmarkY3test.cbor-outlines.cbor"
    # x = parse_davinci_response_file()

    # parse_davinci_as_query_with_full_paragraph_list()
    # print(x[0], x[0].isSection())
    parse_davinci_as_query_with_full_paragraph_list(page_davinci_path=page_davinci_path, section_davinci_path=section_davinci_path, car_outlines_path=car_outlines_path)


def main():
    """Convert Davinci3 responses into paragraphs for grading and evaluation with EXAM."""

    import argparse

    desc = r'''Convert Davinci3 responses into paragraphs for grading and evaluation with EXAM. \n
               \n  
               The input file will be in the following JSONL format (per section or per page): \n
                        { \n
                        "datatime": "Tue Feb 28 11:38:51 EST 2023",\n
                        "response": "\n\nWater Speed and Erosion\n\nWater speed is an important factor in the erosion ...",
                        "sectionQueryStr": "water speed and erosion", \n 
                        "gptmodel": "text-davinci-003", \n
                        "prompt": "Generate a Wikipedia section on \"water speed and erosion\" for an article on \"erosion and deposition by flowing water\".", \n
                        "benchmark": "benchmarkY3test", \n
                        "queryId": "T_0022", \n
                        "queryStr": "erosion and deposition by flowing water", \n
                        "pageOrSection": "section" \n
                        } \n
                    \n
              The output file for exam-grading will be a *JSONL.GZ file that follows this structure: \n
              \n  
                  [query_id, [FullParagraphData]] \n
              \n
               where `FullParagraphData` meets the following structure \n
             ''' + FullParagraphData.schema_json(indent=2)
    
    parser = argparse.ArgumentParser(description="Convert Davinci2 for EXAM scoring"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-p','--davinci-page-file', type=str, metavar='DAVINCI_PAGE_FILE'
                        , help='Input jsonl file for page-level Davinci3 responses'
                        )
    parser.add_argument('-s','--davinci-section-file', type=str, metavar='DAVINCI_SECTION_FILE'
                        , help='Input jsonl file for section-level Davinci3 responses'
                        )
    parser.add_argument('-c','--car-outlines-cbor', type=str, metavar='CAR_OUTLINES_CBOR'
                        , help='Input TREC CAR ourlines file (from which page-level queries and order of sections/facets will be taken)'
                        )
    parser.add_argument('--run', type=str, metavar='FILE'
                        , help='trec-eval compatible run file to produce'
                        )


    parser.add_argument('-o', '--out-file', type=str, metavar='runs-xxx.jsonl.gz', required=True
                        , help='Output file name where paragraphs with exam grade annotations will be written to')

    parser.add_argument('--max-queries', type=int, metavar='INT', default=None, help='limit the number of queries that will be processed (for debugging)')
    parser.add_argument('--simple', action='store_true', help="Will only use the page file")
 

    # Parse the arguments
    args = parser.parse_args()  


    if args.simple:
        contents = simple_davinci_to_full_paragraph_list(page_davinci_path=args.davinci_page_file
                                          , max_queries = args.max_queries
                                          , run_file = args.run)

    else:
        contents = parse_davinci_as_query_with_full_paragraph_list(
                            page_davinci_path=args.davinci_page_file
                            ,section_davinci_path=args.davinci_section_file
                            , car_outlines_path=args.car_outlines_cbor
                            , max_queries = args.max_queries
                            )

    data_model.writeQueryWithFullParagraphs(args.out_file, contents)

if __name__ == "__main__":
    main()

