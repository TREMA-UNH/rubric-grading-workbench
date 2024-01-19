from pydantic import BaseModel
from typing import List, Any, Optional, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import gzip
import json
from pathlib import Path



from .pydantic_helper import pydantic_dump

class SelfRating(BaseModel):
    question_id:str
    self_rating:int


class ExamGrades(BaseModel):
    correctAnswered: List[str]               # [question_id]
    wrongAnswered: List[str]                 # [question_id]
    answers: List[Tuple[str, str]]           # [ [question_id, answer_text]] 
    llm: str                                 # huggingface model name
    llm_options: Dict[str,Any]               # anything that seems relevant
    exam_ratio: float                        # correct / all questions
    prompt_info: Optional[Dict[str,Any]]     # more info about the style of prompting
    self_ratings: Optional[List[SelfRating]] # if availabel: self-ratings (question_id, rating)

    def self_ratings_as_iterable(self):
        if self.self_ratings is None:
            return []
        else:
            return self.self_ratings

@dataclass
class GradeFilter():
    model_name: Optional[str]
    prompt_class: Optional[str]
    is_self_rated: Optional[bool]
    min_self_rating: Optional[int]
    question_set:Optional[str]

    @staticmethod
    def noFilter():
        return GradeFilter(model_name=None, prompt_class=None, is_self_rated=None, min_self_rating=None, question_set=None)

    def filter(self, grade:ExamGrades)-> bool:
        # Note, the following code is based on inverse logic -- any grade that DOES NOT meet set filter requirements is skipped

        # grades are marked as using this model
        if self.model_name is not None:
            if not grade.llm == self.model_name:
                return False
            elif not self.model_name == "google/flan-t5-large":
                return False

        # grade.prompt_info is marked as using this prompt_class
        if self.prompt_class is not None:
            if grade.prompt_info is not None:
                grade_prompt_class = grade.prompt_info.get("prompt_class", None)
                if grade_prompt_class is not None:
                    if not grade_prompt_class == self.prompt_class:
                        return False
            elif not self.prompt_class == "QuestionPromptWithChoices":  # handle case before we tracked prompt info
                return False


        # grade.prompt_info is marked as is_self_rated
        if self.is_self_rated is not None:
            if grade.prompt_info is not None:
                grade_is_self_rated = grade.prompt_info.get("is_self_rated", None)
                if grade_is_self_rated is not None:
                    if not grade_is_self_rated == self.is_self_rated:
                        return False

        # for at least one question, the self_rating is at least self.min_self_rating
        if self.min_self_rating is not None:
            if grade.self_ratings is not None and len(grade.self_ratings)>0:  # grade has self_ratings
                if not any( (rating.self_rating >= self.min_self_rating  for rating in grade.self_ratings) ):
                    return False

        if self.question_set is not None:
            if self.question_set == "tqa":
                is_tqa_question = grade.answers[0][0].startswith("NDQ_")
                if not is_tqa_question:
                    return False
                
            if self.question_set == "naghmeh":
                is_naghmeh_question = grade.answers[0][0].startswith("tqa2:")
                if not is_naghmeh_question:
                    return False

        return True

    def get_min_grade_filter(self, min_self_rating:int):
        return GradeFilter(model_name=self.model_name
                           , prompt_class=self.prompt_class
                           , is_self_rated=self.is_self_rated
                           , min_self_rating=min_self_rating
                           , question_set=self.question_set)

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


class FullParagraphData(BaseModel):
    paragraph_id : str
    text : str
    paragraph : Any
    paragraph_data : ParagraphData
    exam_grades : Optional[List[ExamGrades]]

    def retrieve_exam_grade(self, grade_filter:GradeFilter) -> List[ExamGrades]:
        if self.exam_grades is None:
            return []
        
        found = next((g for g in self.exam_grades if grade_filter.filter(g)), None)
        if found is not None:
            return [found]
        else: 
            return []
        


    def exam_grades_iterable(self)-> List[ExamGrades]:
        return [] if self.exam_grades is None else self.exam_grades

       
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
    # print(line)
    data = json.loads(line)
    return QueryWithFullParagraphList(data[0], [FullParagraphData.parse_obj(paraInfo) for paraInfo in data[1]])


# Path to the benchmarkY3test-qrels-with-text.jsonl.gz file
def parseQueryWithFullParagraphs(file_path:Path) -> List[QueryWithFullParagraphList] :
    '''Load JSONL.GZ file with exam annotations in FullParagraph information'''
    # Open the gzipped file

    result:List[QueryWithFullParagraphList] = list()
    try: 
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            # return [parseQueryWithFullParagraphList(line) for line in file]
            for line in file:
                result.append(parseQueryWithFullParagraphList(line))
    except  EOFError as e:
        print("Warning: Gzip EOFError on {file_path}. Use truncated data....\nFull Error:\n{e}")
    return result



def dumpQueryWithFullParagraphList(queryWithFullParagraph:QueryWithFullParagraphList)->str:
    '''Write `QueryWithFullParagraphList` to jsonl.gz'''
    return  json.dumps ([queryWithFullParagraph.queryId,[p.dict() for p in queryWithFullParagraph.paragraphs]])+"\n"

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

