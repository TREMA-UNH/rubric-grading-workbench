
from typing import List, Dict, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
import re

from .data_model import *
from .data_model import parseQueryWithFullParagraphs, QueryWithFullParagraphList, GradeFilter


from pathlib import Path


@dataclass
class QrelEntry:
    query_id:str
    paragraph_id:str
    grade:int

    def format_qrels(self):
        return ' '.join([self.query_id, '0', self.paragraph_id, str(self.grade)])+'\n'

def exam_to_qrels_files(exam_input_file:Path, qrel_out_file:Path, grade_filter:GradeFilter):
    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)
    qrel_entries = conver_exam_to_qrels(query_paragraphs,grade_filter=grade_filter)
   
    write_qrel_file(qrel_out_file, qrel_entries)

def write_qrel_file(qrel_out_file, qrel_entries):
    '''Use to write qrels file'''
    with open(qrel_out_file, 'wt', encoding='utf-8') as file:
        file.writelines(entry.format_qrels() for entry in qrel_entries)
        file.close()

def conver_exam_to_qrels(query_paragraphs:List[QueryWithFullParagraphList], grade_filter:GradeFilter)->List[QrelEntry]:
    '''workhorse to convert exam-annotated paragraphs into qrel entries.
    load input file with `parseQueryWithFullParagraphs`
    write qrels file with `write_qrel_file`
    or use convenience function `exam_to_qrels_file`
    '''
    qrel_entries:List[QrelEntry] = list()
    
    for queryWithFullParagraphList in query_paragraphs:
        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs

        for para in paragraphs:
            if para.exam_grades:
                for exam_grade in para.retrieve_exam_grade_any(grade_filter=grade_filter): # there will be 1 or 0
                    numCorrect = len(exam_grade.correctAnswered)
                    qrel_entries.append(QrelEntry(query_id=query_id, paragraph_id=para.paragraph_id, grade=numCorrect))
    return qrel_entries

def convert_exam_to_facet_qrels(query_paragraphs:List[QueryWithFullParagraphList], grade_filter:GradeFilter, query_facets: Dict[str,Set[str]]=dict())->List[QrelEntry]:
    '''workhorse to convert exam-annotated paragraphs into qrel entries.
    load input file with `parseQueryWithFullParagraphs`
    write qrels file with `write_qrel_file`
    or use convenience function `exam_to_qrels_file`
    '''
    qrel_entries:List[QrelEntry] = list()
    beforeLastSlashpattern = r"^(.+)/"
    
    def count_by_facet(correctAnswered:List[str])->Dict[str,int]:
        grouped:Dict[str,int] = defaultdict(int) # default: 0
        for question_id in correctAnswered:
            if question_id.startswith("NDQ"):
                # tqa question ID, not facet specific
                for facet in query_facets[query_id]:
                    grouped[f"{query_id}/{facet}"]+=1
            else:
                match = re.search(beforeLastSlashpattern, question_id)            
                if match:
                    facet_id = match.group(1)
                    grouped[facet_id]+=1
        return grouped
    
    for queryWithFullParagraphList in query_paragraphs:
        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs

        for para in paragraphs:
            if para.exam_grades:
                for exam_grade in para.retrieve_exam_grade_any(grade_filter=grade_filter): # there will be 1 or 0
                    grouped:Dict[str,int] = count_by_facet(exam_grade.correctAnswered)
                    for facet_id, count in grouped.items():
                        qrel_entries.append(QrelEntry(query_id=facet_id, paragraph_id=para.paragraph_id, grade=count))
                    
    return qrel_entries

def convert_direct_to_facet_qrels(query_paragraphs:List[QueryWithFullParagraphList], grade_filter:GradeFilter,query_facets: Dict[str,Set[str]]=dict())->List[QrelEntry]:
    '''workhorse to convert exam-annotated paragraphs into qrel entries.
    load input file with `parseQueryWithFullParagraphs`
    write qrels file with `write_qrel_file`
    or use convenience function `exam_to_qrels_file`
    '''
    qrel_entries:List[QrelEntry] = list()
    beforeLastSlashpattern = r"^(.+)/"
    

    for queryWithFullParagraphList in query_paragraphs:
        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs

        for para in paragraphs:

            if para.grades:
                for grade in para.retrieve_exam_grade_any(grade_filter=grade_filter): # there will be 1 or 0
                    for facet_id in query_facets[query_id]:
                        qrel_entries.append(QrelEntry(query_id=f"{query_id}/{facet_id}", paragraph_id=para.paragraph_id, grade=1 if grade.correctAnswered else 0))
                    
    return qrel_entries


def convert_direct_to_rated_facet_qrels(query_paragraphs:List[QueryWithFullParagraphList], grade_filter:GradeFilter,query_facets: Dict[str,Set[str]]=dict())->List[QrelEntry]:
    '''workhorse to convert exam-annotated paragraphs into qrel entries.
    load input file with `parseQueryWithFullParagraphs`
    write qrels file with `write_qrel_file`
    or use convenience function `exam_to_qrels_file`
    '''
    qrel_entries:List[QrelEntry] = list()
    beforeLastSlashpattern = r"^(.+)/"
    

    for queryWithFullParagraphList in query_paragraphs:
        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs

        for para in paragraphs:
            if para.grades:
                for grade in para.retrieve_exam_grade_any(grade_filter=grade_filter): # there will be 1 or 0
                    for facet_id in query_facets[query_id]:
                        if grade.self_ratings and len(grade.self_ratings)>0:
                            # we have self-ratings
                            qrel_entries.append(QrelEntry(query_id=f"{query_id}/{facet_id}", paragraph_id=para.paragraph_id, grade=grade.self_ratings[0].self_rating))
                        else:
                            # fall back to boolean grade
                            qrel_entries.append(QrelEntry(query_id=f"{query_id}/{facet_id}", paragraph_id=para.paragraph_id, grade=1 if grade.correctAnswered else 0))    
    return qrel_entries


def convert_exam_to_rated_facet_qrels(query_paragraphs:List[QueryWithFullParagraphList], grade_filter:GradeFilter, query_facets: Dict[str,Set[str]]=dict())->List[QrelEntry]:
    '''workhorse to convert exam-annotated paragraphs into qrel entries.
    load input file with `parseQueryWithFullParagraphs`
    write qrels file with `write_qrel_file`
    or use convenience function `exam_to_qrels_file`
    '''
    qrel_entries:List[QrelEntry] = list()
    beforeLastSlashpattern = r"^(.+)/"
    
    def best_rating_by_facet(self_ratings:List[SelfRating])->Dict[str,int]:
        grouped:Dict[str,Set[int]] = defaultdict(set) # default: 0
        for self_rating in self_ratings:
            question_id = self_rating.get_id()
            rating = self_rating.self_rating

            if question_id.startswith("NDQ"):
                # tqa question ID, not facet specific
                for facet in query_facets[query_id]:
                    grouped[f"{query_id}/{facet}"].add(rating)
            else:
                # question bank, parse out the facet

                match = re.search(beforeLastSlashpattern, question_id)            
                if match:
                    facet_id = match.group(1)
                    grouped[facet_id].add(rating)
        best_rating = {facet_id:max(ratings)  for facet_id, ratings in grouped.items()}
        return best_rating

   
    def count_by_facet(correctAnswered:List[str])->Dict[str,int]:
        grouped:Dict[str,int] = defaultdict(int) # default: 0
        for question_id in correctAnswered:

            if question_id.startswith("NDQ"):
                # tqa question ID, not facet specific
                for facet in query_facets[query_id]:
                    grouped[f"{query_id}/{facet}"]+=1
            else:
                # question bank, parse out the facet

                match = re.search(beforeLastSlashpattern, question_id)            
                if match:
                    facet_id = match.group(1)
                    grouped[facet_id]+=1
        return grouped


    for queryWithFullParagraphList in query_paragraphs:
        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs

        for para in paragraphs:
            if para.exam_grades:
                for exam_grade in para.retrieve_exam_grade_any(grade_filter=grade_filter): # there will be 1 or 0
                    if exam_grade.self_ratings is None:
                        # Fall back on binary grades
                        # raise RuntimeError(f"Qrels from self-ratings asked on exam grades without self-ratings.\ngrade_filter {grade_filter}\nOffending grade {exam_grade}")
                        grouped:Dict[str,int] = count_by_facet(exam_grade.correctAnswered)
                        for facet_id, count in grouped.items():
                            qrel_entries.append(QrelEntry(query_id=facet_id, paragraph_id=para.paragraph_id, grade=count))

                    else:
                        best_rating:Dict[str,int] = best_rating_by_facet(exam_grade.self_ratings)
                        for facet_id, rating in best_rating.items():
                            qrel_entries.append(QrelEntry(query_id=facet_id, paragraph_id=para.paragraph_id, grade=rating))

    return qrel_entries

def convert_exam_to_rated_qrels(query_paragraphs:List[QueryWithFullParagraphList], grade_filter:GradeFilter)->List[QrelEntry]:
    '''workhorse to convert exam-annotated paragraphs into qrel entries.
    load input file with `parseQueryWithFullParagraphs`
    write qrels file with `write_qrel_file`
    or use convenience function `exam_to_qrels_file`
    '''
    qrel_entries:List[QrelEntry] = list()
    beforeLastSlashpattern = r"^(.+)/"
    
    def best_rating_by_query(self_ratings:List[SelfRating])->int:
        best_rating = max([rating.self_rating for rating in self_ratings])
        return best_rating
    
    for queryWithFullParagraphList in query_paragraphs:
        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs

        for para in paragraphs:
            if para.exam_grades:
                for exam_grade in para.retrieve_exam_grade_any(grade_filter=grade_filter): # there will be 1 or 0
                    if exam_grade.self_ratings is None:
                        raise RuntimeError(f"Qrels from self-ratings asked on exam grades without self-ratings.\ngrade_filter {grade_filter}\nOffending grade {exam_grade}")
                    best_rating:int = best_rating_by_query(exam_grade.self_ratings)
                    qrel_entries.append(QrelEntry(query_id=query_id, paragraph_id=para.paragraph_id, grade=best_rating))
    return qrel_entries

def main():
    import argparse

    desc = f'''Convert paragraphs with exam annotations to a paragraph qrels file. \n
              The judgment level is the number of correctly answerable questions. \n
              The input file (i.e, exam_annotated_file) has to be a *JSONL.GZ file that follows this structure: \n
              \n  
                  [query_id, [FullParagraphData]] \n
              \n
               where `FullParagraphData` meets the following structure \n
             {FullParagraphData.schema_json(indent=2)}
             '''
    

    # Create the parser
    # parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser = argparse.ArgumentParser(description="Convert paragraphs with exam annotations to a paragraph qrels file."
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exam_annotated_file', type=str, metavar='exam-xxx.jsonl.gz'
                        , help='json file that annotates each paragraph with a number of anserable questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )

    # Add an optional output file argument
    parser.add_argument('-o', '--output', type=str, metavar="FILE", help='Output QREL file name', default='output.qrels')
    parser.add_argument('-m', '--model', type=str, metavar="HF_MODEL_NAME", help='the hugging face model name used by the Q/A module.')

    # Parse the arguments
    args = parser.parse_args()    
    exam_to_qrels_files(exam_input_file=args.exam_annotated_file, qrel_out_file=args.output, model_name=args.model)


if __name__ == "__main__":
    main()
