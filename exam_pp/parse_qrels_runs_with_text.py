from pydantic import BaseModel
from typing import List, Any, Optional, Dict, Tuple
from dataclasses import dataclass

import gzip
import json
import itertools
from pathlib import Path

class Judgment(BaseModel):
    paragraphId : str
    query : str
    relevance : int
    titleQuery : str
    

class ParagraphRankingEntry(BaseModel):
    method : str
    paragraphId : str
    queryId : str # can be title query or query facet, e.g. "tqa2:L_0002/T_0020"
    rank : int
    score : float
    

class ParagraphData(BaseModel):
    judgments : List[Judgment]
    rankings : List[ParagraphRankingEntry] 





class ExamGrades(BaseModel):
    correctAnswered: List[str]      # [question_id]
    wrongAnswered: List[str]        # [question_id]
    answers: List[Tuple[str, str]]  # [ [question_id, answer_text]] 
    llm: str                        # huggingface model name
    llm_options: Dict[str,Any]      # anything that seems relevant
    exam_ratio: float               # correct / all questions

class FullParagraphData(BaseModel):
    paragraph_id : str
    text : str
    paragraph : Any
    paragraph_data : ParagraphData
    exam_grades : Optional[List[ExamGrades]]

    def exam_grades_iterable(self)-> List[ExamGrades]:
        return [] if self.exam_grades is None else self.exam_grades

    def get_any_exam_grade(self)->Optional[ExamGrades]:
        if self.exam_grades is None or len(self.exam_grades)<1: 
            return None
        else: 
            return self.exam_grades[0]
        
    def get_any_judgment(self)->Optional[Judgment]:
        if self.paragraph_data.judgments is None or len(self.paragraph_data.judgments)<1: 
            return None
        else: 
            return self.paragraph_data.judgments[0]

    def get_any_ranking(self, method_name:str)->Optional[ParagraphRankingEntry]:
        if self.paragraph_data.rankings is None or len(self.paragraph_data.rankings)<1: 
            return None
        else:
            return next((item for item in self.paragraph_data.rankings if item.method==method_name), None)

@dataclass
class QueryWithFullParagraphList():
    queryId:str
    paragraphs: List[FullParagraphData]


def parseQueryWithFullParagraphList(line:str) -> QueryWithFullParagraphList:
    # Parse the JSON content of the line
    data = json.loads(line)
    return QueryWithFullParagraphList(data[0], [FullParagraphData.parse_obj(paraInfo) for paraInfo in data[1]])


# Path to the benchmarkY3test-qrels-with-text.jsonl.gz file
def parseQueryWithFullParagraphs(file_path:Path) -> [QueryWithFullParagraphList] :
    '''Load JSONL.GZ file with exam annotations in FullParagraph information'''
    # Open the gzipped file
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        return [parseQueryWithFullParagraphList(line) for line in file]


def dumpQueryWithFullParagraphList(queryWithFullParagraph:QueryWithFullParagraphList)->str:
    '''Write `QueryWithFullParagraphList` to jsonl.gz'''
    return  json.dumps ([queryWithFullParagraph.queryId,[p.dict() for p in queryWithFullParagraph.paragraphs]])

def writeQueryWithFullParagraphs(file_path:Path, queryWithFullParagraphList:List[QueryWithFullParagraphList]) :
    # Open the gzipped file
    with gzip.open(file_path, 'wt', encoding='utf-8') as file:
        # Iterate over each line in the file
        file.writelines([dumpQueryWithFullParagraphList(x) for x in queryWithFullParagraphList])



def main():
    """Entry point for the module."""
    x = parseQueryWithFullParagraphs("./benchmarkY3test-qrels-with-text.jsonl.gz")
    print(x[0])

if __name__ == "__main__":
    main()

