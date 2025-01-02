from collections import defaultdict
import gzip
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from pydantic.v1 import BaseModel

from exam_pp import query_loader, data_model
from exam_pp.exam_to_qrels import QrelEntry 
from exam_pp.data_model import *

QREL_IMPORT_PROMPT_CLASS = "QrelImport"

def read_qrel_file(qrel_in_file:Path) ->List[QrelEntry]:
    '''Use to read qrel file'''
    with open(qrel_in_file, 'rt') as file:
        qrel_entries:List[QrelEntry] = list()
        for line in file.readlines():
            splits = line.split(" ")
            qrel_entry=None
            if len(splits)>=4:
                qrel_entry = QrelEntry(query_id=splits[0].strip(), paragraph_id=splits[2].strip(), grade=int(splits[3].strip()))
            elif len(splits)>=3: # we have a qrels file to complete.
                qrel_entry = QrelEntry(query_id=splits[0].strip(), paragraph_id=splits[2].strip(), grade=-99)
            else:
                raise RuntimeError(f"All lines in qrels file needs to contain four columns, or three for qrels to be completed. Offending line: \"{line}\"")
            
            qrel_entries.append(qrel_entry)
            # print(f"{line}\n {qrel_entry}")
    return qrel_entries

def read_query_file(query_file:Path, max_queries:Optional[int]=None) -> Dict[str,str]:
    with open(query_file, 'rt') as file:
        query_dict = dict()
        for line in itertools.islice(file.readlines(), max_queries):
            splits = line.split("\t")
            if len(splits)>=2:
                query_dict[splits[0].strip()]=splits[1].strip()
            else:
                raise RuntimeError(f"each line in query file {query_file} must contain two tab-separated columns. Offending line: \"{line}\"")

    return query_dict



# class LLMJudgeDocument(BaseModel):
#     docid:str
#     doc:str

# def parseLLMJudgeDocument(line:str) -> LLMJudgeDocument:
#     # Parse the JSON content of the line
#     # print(line)
#     return LLMJudgeDocument.parse_raw(line)

# def loadLLMJudgeCorpus(file_path:Path, max_paragraphs:Optional[int]) -> List[LLMJudgeDocument]:
#     '''Load LLMJudge document corpus'''

#     result:List[LLMJudgeDocument] = list()
#     try: 
#         with open(file_path, 'rt', encoding='utf-8') as file:
#             # return [parseQueryWithFullParagraphList(line) for line in file]
#             for line in itertools.islice(file.readlines(), max_paragraphs):
#                 result.append(parseLLMJudgeDocument(line))
#     except  EOFError as e:
#         print(f"Warning: File EOFError on {file_path}. Use truncated data....\nFull Error:\n{e} \n offending line: \n {line}")
#     return result

# def write_query_file(file_path:Path, queries:Dict[str,str])->None:
#     with open(file_path, 'wt', encoding='utf-8') as file:
#         json.dump(obj=queries,fp=file)



def convert_paragraphs(input_qrels_by_qid:Dict[str,List[QrelEntry]]
                       , rubric_data:List[QueryWithFullParagraphList]
                       , query_str_by_qid:Dict[str,str]
                       , qrel_as_judgment:bool
                       , qrel_as_grade:bool
                       , append_judgment:bool
                       , import_options:Dict[str,Any]=dict()
                       )->List[QueryWithFullParagraphList]:

    for entry in rubric_data:
        query_id = entry.queryId
        query_str = query_str_by_qid.get(query_id)
        if query_str is None:
            print(f"Can't identify title query_str for query {query_id}, using empty string")
            query_str = ""

        qrel_entries = {qentry.paragraph_id: qentry for qentry in  input_qrels_by_qid[query_id]}
        for para in entry.paragraphs:
            qrel_entry = qrel_entries.get(para.paragraph_id)

            if qrel_entry is not None:
                # add as judgment
                if qrel_as_judgment:
                    judgment = Judgment(paragraphId= qrel_entry.paragraph_id, query=query_id, relevance=qrel_entry.grade, titleQuery=query_str)
                    if append_judgment:
                        para.paragraph_data.judgments.append(judgment)
                    else:
                        para.paragraph_data.judgments=[judgment]

                # add as grades
                if qrel_as_grade:
                    if para.grades is None:
                        para.grades = list()
                    para.grades.append(Grades(correctAnswered= qrel_entry.grade>0
                                            , answer=f"{qrel_entry.grade}"
                                            , llm="imported"
                                            , llm_options={}
                                            , prompt_info=import_options
                                            , self_ratings=qrel_entry.grade
                                            , relevance_label=qrel_entry.grade
                                            , prompt_type=DIRECT_GRADING_PROMPT_TYPE
                                            ))

    return rubric_data

def main(cmdargs=None):
    """Convert LLMJudge data to inputs for EXAM/RUBRIC."""

    import argparse

    desc = f'''Convert LLMJudge data to inputs for EXAM/RUBRIC. \n
              The RUBRIC input will to be a *JSONL.GZ file.  Info about JSON schema with --help-schema
             '''
    help_schema=f'''The input and output file (i.e, exam_annotated_file) has to be a *JSONL.GZ file that follows this structure: \n
                \n  
                    [query_id, [FullParagraphData]] \n
                \n
                where `FullParagraphData` meets the following structure \n
                {FullParagraphData.schema_json(indent=2)}
                \n
                Create a compatible file with 
                exam_pp.data_model.writeQueryWithFullParagraphs(file_path:Path, queryWithFullParagraphList:List[QueryWithFullParagraphList])
                '''
            
    parser = argparse.ArgumentParser(description="Convert TREC Qrels data to RUBRIC judgments or grades."
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter
                                   )
    parser.add_argument('rubric_in', type=str, metavar='xxx.jsonl.gz'
                        , help='input RUBRIC file in jsonl.gz format'
                        )


    # parser.add_argument('llmjudge_corpus', type=str, metavar='xxx.jsonl.gz'
    #                     , help='input json file with corpus from the LLMJudge collection'
    #                     )


    parser.add_argument('-o', '--output', type=str, metavar='xxx.jsonl.gz'
                        , help='output path for RUBRIC file in jsonl.gz format.'
                        )

    parser.add_argument('--query-path', type=str, metavar='PATH', help='Path to read LLMJudge queries')
    parser.add_argument('--input-qrel-path', type=str, metavar='PATH', help='Path to read LLMJudge qrels (to be completed)')
    # parser.add_argument('--query-out', type=str, metavar='PATH', help='Path to write queries for RUBRIC/EXAM to')

    parser.add_argument('--as-grade',  action='store_true',  help='store qrel info as grade')
    parser.add_argument('--prompt-class',  type=str,  help=f'if set, will be used as prompt-class name, otherwise it is set to {QREL_IMPORT_PROMPT_CLASS}')
    parser.add_argument('--as-judgment',  action='store_true',  help='store qrel info as judgment (i.e., the ground truth)')
    parser.add_argument('--append-judgment',  action='store_true',  help='Instead of replacing the judgment, append it as an alternative judgment')
    

    parser.add_argument('--max-queries', type=int, metavar='INT', default=None, help='limit the number of queries that will be processed (for debugging)')
    parser.add_argument('--max-paragraphs', type=int, metavar='INT', default=None, help='limit the number of paragraphs that will be processed (for debugging)')


    parser.add_argument('--help-schema', action='store_true', help="Additional info on required JSON.GZ input format")

    # Parse the arguments
    args = parser.parse_args(args = cmdargs)  

    if args.help_schema:
        print(help_schema)
        sys.exit()


    # First we load all queries
    # query_set:Dict[str,str] 
    query_set = read_query_file(query_file=args.query_path, max_queries = args.max_queries)

    # Fetch the qrels file  ... and munge
    input_qrels = read_qrel_file(qrel_in_file=args.input_qrel_path)
    qrel_query_ids = {q.query_id  for q in input_qrels}
    input_qrels_by_qid:Dict[str,List[QrelEntry]] = defaultdict(list)
    for qrel_entry in input_qrels:
        input_qrels_by_qid[qrel_entry.query_id].append(qrel_entry)
        # print(f"{qrel_entry}")

    

    # print(f"query_set = {query_set}")

    # load the paragraph data
    # corpus = loadLLMJudgeCorpus(file_path = args.llmjudge_corpus, max_paragraphs = args.max_paragraphs)
    # corpus_by_para_id = {para.docid: para  for para in corpus}

    # print(f"corpus = {corpus}")
    

    rubric_in_file:List[QueryWithFullParagraphList] 
    rubric_in_file = parseQueryWithFullParagraphs(file_path=args.rubric_in)

    import_options = {"prompt_class": args.prompt_class or QREL_IMPORT_PROMPT_CLASS
                      , "qrels_file:": args.input_qrel_path
                      }

    # now emit the input files for RUBRIC/EXAM
    rubric_data:List[QueryWithFullParagraphList] 
    rubric_data = convert_paragraphs(input_qrels_by_qid=input_qrels_by_qid
                                                                      , rubric_data=rubric_in_file
                                                                      , query_str_by_qid=query_set
                                                                      , qrel_as_grade=args.as_grade
                                                                      , qrel_as_judgment= args.as_judgment
                                                                      , append_judgment=args.append_judgment
                                                                      , import_options = import_options )
 

    writeQueryWithFullParagraphs(args.output, queryWithFullParagraphList=rubric_data)



if __name__ == "__main__":
    main()
