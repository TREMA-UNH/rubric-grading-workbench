from collections import defaultdict
import itertools
from pydantic import BaseModel
from typing import Iterable, List, Any, Optional, Dict, Tuple, Union, cast
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import pathlib

from .test_bank_prompts import DirectGradingPrompt



from .pydantic_helper import pydantic_dump

class SelfRating(BaseModel):
    question_id:Optional[str]
    nugget_id:Optional[str]=None
    self_rating:int
    def get_id(self)->str:
        if self.question_id is not None:
            return self.question_id
        elif self.nugget_id is not None:
            return self.nugget_id
        else:
            ##   TODO   direct grading prompts will have neither
            raise RuntimeError("Neither question_id nor nugget_id is given.")

    def __hash__(self):
        # Hash a tuple of all field values
        return hash(tuple(self.__dict__.values()))

    def __eq__(self, other):
        if not isinstance(other, SelfRating):
            return False
        # Compare all field values
        return self.__dict__ == other.__dict__

class ExamGrades(BaseModel):
    correctAnswered: List[str]               # [question_id]
    wrongAnswered: List[str]                 # [question_id]
    answers: List[Tuple[str, str]]           # [ [question_id, answer_text]] 
    llm: str                                 # huggingface model name
    llm_options: Dict[str,Any]               # anything that seems relevant
    exam_ratio: float                        # correct / all questions
    prompt_info: Optional[Dict[str,Any]]     # more info about the style of prompting
    self_ratings: Optional[List[SelfRating]] # if availabel: self-ratings (question_id, rating)
    prompt_type: Optional[str]

    def self_ratings_as_iterable(self)->List[SelfRating]:
        if self.self_ratings is None:
            return []
        else:
            return self.self_ratings

    def __hash__(self):
        # Hash a tuple of all field values
        return hash(tuple(self.__dict__.values()))

    def __eq__(self, other):
        if not isinstance(other, ExamGrades):
            return False
        # Compare all field values
        return self.__dict__ == other.__dict__

class Grades(BaseModel):
    correctAnswered: bool               # true if relevant,  false otherwise
    answer: str                        #  llm_response_text
    llm: str                                 # huggingface model name  google/flan-t5-large
    llm_options: Dict[str,Any]               # anything that seems relevant
    prompt_info: Optional[Dict[str,Any]]     # more info about the style of prompting
    self_ratings: Optional[int]         #  if available: self-rating (e.g. 0-5)
    prompt_type: Optional[str]

    def self_ratings_as_iterable(self):
        if self.self_ratings is None:
            return []
        else:
            return [self.self_ratings]

    def __hash__(self):
        # Hash a tuple of all field values
        return hash(tuple(self.__dict__.values()))

    def __eq__(self, other):
        if not isinstance(other, Grades):
            return False
        # Compare all field values
        return self.__dict__ == other.__dict__
        # must have fields:
        #  prompt_info["prompt_class"]="FagB"
            # info =  {
            #       "prompt_class":  Fag  # or self.__class__.__name__
            #     , "prompt_style":  old_prompt("prompt_style", "question-answering prompt")
            #     , "is_self_rated": false # false if not self-rated, otherwise true
            #     }

    def as_exam_grades(self)-> ExamGrades:

        if self.correctAnswered:
            correctAnswered = ["direct"]
            wrongAnswered = []
            exam_ratio=1.0
        else:
            wrongAnswered = ["direct"]
            correctAnswered = []
            exam_ratio=0.0

        self_ratings = None
        if self.self_ratings is not None:
            self_ratings = [SelfRating(question_id="direct",nugget_id="direct",self_rating=self.self_ratings)]

        return ExamGrades(correctAnswered=correctAnswered
                          , wrongAnswered=wrongAnswered
                          , answers=[("direct",self.answer)]
                          , exam_ratio=exam_ratio
                          , self_ratings=self_ratings
                          , llm=self.llm
                          , llm_options=self.llm_options
                          , prompt_info=self.prompt_info
                          , prompt_type=DirectGradingPrompt.my_prompt_type
        )
@dataclass
class GradeFilter():
    model_name: Optional[str]
    prompt_class: Optional[str]
    is_self_rated: Optional[bool]
    min_self_rating: Optional[int]
    question_set:Optional[str]
    prompt_type:Optional[str]

    def print_name(self)->str:
        return f"{self.prompt_class} {self.model_name} {self.question_set}"

    @staticmethod
    def noFilter():
        return GradeFilter(model_name=None, prompt_class=None, is_self_rated=None, min_self_rating=None, question_set=None, prompt_type=None)

    @staticmethod
    def question_type(grade:Union[ExamGrades,Grades]):
        if isinstance(grade, ExamGrades):
            if (grade.answers[0][0].startswith("NDQ_")):
                return "tqa"
            elif (grade.answers[0][0].startswith("tqa2:")):
                return "genq"
            else: 
                return "question-bank"
        else:
            "direct"


    @staticmethod
    def get_prompt_type(grade:Union[ExamGrades,Grades])->str:
        return grade.prompt_type if grade.prompt_type is not None else ""

        

    @staticmethod
    def key_dict(grade:Union[ExamGrades,Grades])->Dict[str,Any]:
        return  { "llm": grade.llm
                , "prompt_class": grade.prompt_info.get("prompt_class", None) if grade.prompt_info is not None else "QuestionPromptWithChoices"
                , "is_self_rated": grade.prompt_info.get("is_self_rated", None) if grade.prompt_info is not None else False
                , "question_set": GradeFilter.question_type(grade)
                , "prompt_type": GradeFilter.get_prompt_type(grade)
                }
    
    @staticmethod
    def key(grade:Union[ExamGrades,Grades])->str:
        is_self_rated = grade.prompt_info.get("is_self_rated", None) if grade.prompt_info is not None else False
        prompt_class =grade.prompt_info.get("prompt_class", None) if grade.prompt_info is not None else "QuestionPromptWithChoices"
        prompt_type = GradeFilter.get_prompt_type(grade)
        return f"llm={grade.llm} prompt_class={prompt_class} is_self_rated={is_self_rated}  question_set={GradeFilter.question_type(grade)} prompt_type={prompt_type}"
                

    def fetch_any(self, exam_grades: Optional[List[ExamGrades]], grades: Optional[List[Grades]])-> List[ExamGrades]:
        gs = [cast(ExamGrades, g) for g in self.fetch(exam_grades)]
        gs2 = [cast(Grades, g).as_exam_grades() for g in self.fetch(grades)]
        gs.extend( gs2 )
        return gs


    def fetch(self, grades:Optional[Union[List[ExamGrades],List[Grades]]])-> List[Union[ExamGrades,Grades]]:
        if grades is None:
            return []
        else:
            res=  [ g for g in grades if self.filter(g) ]
            return res
    
    def filter_grade(self, grade:Grades)-> bool:
        return self.filter(grade)

    def filter(self, grade:Union[ExamGrades,Grades])-> bool:
        # Note, the following code is based on inverse logic -- any grade that DOES NOT meet set filter requirements is skipped

        # grades are marked as using this model
        if self.model_name is not None:
            if grade.llm is None:  # old run, where we did not expose this was a flan-t5-large run
                if self.model_name == "google/flan-t5-large":
                    pass # this is acceptable
                else:
                    return False  # we are asking for a different model

            if not grade.llm == self.model_name:  # grade.llm is set, so lets see whether it matches
                return False    # sad trombone, did not match

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
        print(f"gradefilter is self_rated:{self.is_self_rated}  grade.self_ratings={grade.self_ratings}")
        if self.is_self_rated is not None:
            if self.is_self_rated and grade.self_ratings is not None:
                pass
            elif not self.is_self_rated and grade.self_ratings is None:
                pass
            else:
                return False 


            # if grade.prompt_info is not None:
            #     grade_is_self_rated = grade.prompt_info.get("is_self_rated", None)
            #     if grade_is_self_rated is not None:
            #         if not grade_is_self_rated == self.is_self_rated:
            #             return False

        # prompt type (question, nugget, direct)
        if self.prompt_type is not None:
            if grade.prompt_type is not None:
                if not self.prompt_type == grade.prompt_type:
                    return False

        if isinstance(grade, ExamGrades):
            # for at least one question, the self_rating is at least self.min_self_rating
            if self.min_self_rating is not None:
                if grade.self_ratings is not None and len(grade.self_ratings)>0:  # grade has self_ratings
                    if not any( (rating.self_rating >= self.min_self_rating  for rating in grade.self_ratings) ):
                        return False

            if self.question_set is not None:
                is_tqa_question = grade.answers[0][0].startswith("NDQ_")
                is_genq_question = grade.answers[0][0].startswith("tqa2:")

                if self.question_set == "tqa":
                    if not is_tqa_question:
                        return False
                    
                if self.question_set == "genq":
                    if not is_genq_question:
                        return False

                # Todo need a better way to identify the question set. Maybe load from file?
                if self.question_set == "question-bank":
                    if is_tqa_question:
                        return False

                    

        return True

    def get_min_grade_filter(self, min_self_rating:int):
        return GradeFilter(model_name=self.model_name
                           , prompt_class=self.prompt_class
                           , is_self_rated=self.is_self_rated
                           , min_self_rating=min_self_rating
                           , question_set=self.question_set
                           , prompt_type= self.prompt_type
                           )

class Judgment(BaseModel):
    paragraphId : str
    query : str
    relevance : int
    titleQuery : str

    def __hash__(self):
        # Hash a tuple of all field values
        return hash(tuple(self.__dict__.values()))

    def __eq__(self, other):
        if not isinstance(other, Judgment):
            return False
        # Compare all field values
        return self.__dict__ == other.__dict__    

class ParagraphRankingEntry(BaseModel):
    method : str
    paragraphId : str
    queryId : str # can be title query or query facet, e.g. "tqa2:L_0002/T_0020"
    rank : int
    score : float
    
    def __hash__(self):
        # Hash a tuple of all field values
        return hash(tuple(self.__dict__.values()))

    def __eq__(self, other):
        if not isinstance(other, ParagraphRankingEntry):
            return False
        # Compare all field values
        return self.__dict__ == other.__dict__
class ParagraphData(BaseModel):
    judgments : List[Judgment]
    rankings : List[ParagraphRankingEntry] 


class FullParagraphData(BaseModel):
    paragraph_id : str
    text : str
    paragraph : Any
    paragraph_data : ParagraphData
    exam_grades : Optional[List[ExamGrades]]
    grades: Optional[List[Grades]]

    def retrieve_exam_grade_any(self, grade_filter:GradeFilter) -> List[ExamGrades]:
            if self.grades is not None:
                found = grade_filter.fetch_any(self.exam_grades, self.grades)[0]
                if found is not None:
                    return [found]

            if self.exam_grades is not None:
                print("exam grades is not None")
            
                found = next((g for g in self.exam_grades if grade_filter.filter(g)), None)
                if found is not None:
                    return [found]
            
            return []
        

    def retrieve_exam_grade_all(self, grade_filter:GradeFilter) -> List[ExamGrades]:
            found_in_grades = []
            found_in_exam_grades = []
            if self.grades is not None:
                found_in_grades = [g.as_exam_grades() for g in self.grades if grade_filter.filter(g)]
            

            if self.exam_grades is not None:
                found_in_exam_grades = [g for g in self.exam_grades if grade_filter.filter(g)]
        
            return found_in_exam_grades + found_in_grades


    def exam_grades_iterable(self)-> List[ExamGrades]:
        return [] if self.exam_grades is None else self.exam_grades
    
    def grades_iterable(self)-> List[Grades]:
        return [] if self.grades is None else self.grades



    def retrieve_grade_any(self, grade_filter:GradeFilter) -> List[Grades]:
        if self.grades is None:
            return []
        
        found = next((g for g in self.grades if grade_filter.filter_grade(g)), None)
        if found is not None:
            return [found]
        else: 
            return []
        



       
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
        print(f"Warning: Gzip EOFError on {file_path}. Use truncated data....\nFull Error:\n{e}")
    return result



def dumpQueryWithFullParagraphList(queryWithFullParagraph:QueryWithFullParagraphList)->str:
    '''Write `QueryWithFullParagraphList` to jsonl.gz'''
    return  json.dumps ([queryWithFullParagraph.queryId,[p.dict(exclude_none=True) for p in queryWithFullParagraph.paragraphs]])+"\n"

def writeQueryWithFullParagraphs(file_path:Path, queryWithFullParagraphList:List[QueryWithFullParagraphList]) :
    # Open the gzipped file
    with gzip.open(file_path, 'wt', encoding='utf-8') as file:
        # Iterate over each line in the file
        file.writelines([dumpQueryWithFullParagraphList(x) for x in queryWithFullParagraphList])

def unique(elems:Iterable[BaseModel])->List[BaseModel]:
    return list(set(elems))

def merge(files:List[Path], out:Path):
    collection:Dict[str,Dict[str,List[FullParagraphData]]] = defaultdict(lambda : defaultdict(list)) # queryid -> (paraId -> List[FullPargaraphData])
                     
    for infile in files:
        for query_paras in parseQueryWithFullParagraphs(file_path=infile):
            for para in query_paras.paragraphs:
                collection[query_paras.queryId][para.paragraph_id].append(para)

    mergedQueries = list()
    for queryId, data in  collection.items():
        merged_paras:List[FullParagraphData] = list()

        for paraId, paras in data.items():
            p = paras[0]
            merged_para = FullParagraphData(paragraph_id=p.paragraph_id
                                            , text=p.text
                                            , paragraph=p.paragraph
                                            , paragraph_data=ParagraphData(judgments= list(), rankings= list())
                                            , exam_grades=list()
                                            , grades=list())

            grouped_exam_grades:Dict[Any,List[ExamGrades]] = defaultdict(list)
            grouped_grades:Dict[Any,List[Grades]] = defaultdict(list)
            for para in paras:
                # paragraph
                if  merged_para.paragraph is None and para.paragraph is not None:
                    merged_para.paragraph = para.paragraph

                # exam grades
                for eg in para.exam_grades_iterable():
                    key = GradeFilter.key(eg)
                    grouped_exam_grades[key].append(eg)
                
                # grades
                for g in (para.grades if para.grades is not None else list()):
                    key = GradeFilter.key(g)
                    grouped_grades[key].append(g)
                

            for k in grouped_exam_grades.keys():
                print(f"exam_grades key {k}")
            for k in grouped_grades.keys():
                print(f"grades key {k}")

            # exam grades
            merged_para.exam_grades = [gs[0] for gs in grouped_exam_grades.values()]
            # grades
            merged_para.grades = [gs[0] for gs in grouped_grades.values()]

            # judgments
            merged_para.paragraph_data.judgments = list( set(j for p in paras for j in p.paragraph_data.judgments))

            # rankings
            merged_para.paragraph_data.rankings = list( set(r for p in paras for r in p.paragraph_data.rankings))
            merged_paras.append(merged_para)


        qp = QueryWithFullParagraphList(queryId=queryId, paragraphs = merged_paras)
        mergedQueries.append(qp)

    writeQueryWithFullParagraphs(file_path=out, queryWithFullParagraphList=mergedQueries)


def convert(files:List[Path], outdir:Optional[Path], outfile:Optional[Path], ranking_method:Optional[str], grade_llm:Optional[str], old_grading_prompt:Optional[str], grading_prompt:Optional[str]):

    for infile in files:
        converted= list()
        for query_paras in parseQueryWithFullParagraphs(file_path=infile):
            for para in query_paras.paragraphs:
                if ranking_method is not None:
                    for r in para.paragraph_data.rankings:
                        r.method = ranking_method

                if grade_llm is not None:
                    if para.exam_grades:
                        for eg in  para.exam_grades:
                            eg.llm=grade_llm
                    if para.grades:
                        for g in para.grades:
                            g.llm=grade_llm

                if grading_prompt is not None:
                    if para.exam_grades:
                        for eg in para.exam_grades:
                            if (eg.prompt_info is None and old_grading_prompt is None) or (eg.prompt_info is not None and eg.prompt_info.get("prompt_class")==old_grading_prompt):  # the old prompt could also have been set to None.
                                if eg.prompt_info is None:
                                    eg.prompt_info = dict()
                                eg.prompt_info["prompt_class"]=grading_prompt
                    if para.grades:
                        for g in para.grades:
                            if (g.prompt_info is None and old_grading_prompt is None) or (g.prompt_info is not None and g.prompt_info.get("prompt_class")==old_grading_prompt):
                                if g.prompt_info is None:
                                    g.prompt_info = dict()
                                g.prompt_info["prompt_class"]=grading_prompt

            converted.append(query_paras)



        out:Path
        if outdir is None and outfile is None:
            print("overwriting original xxx.jsonl.gz files")
            out = infile
        elif outdir is not None:
            print(f" Writing converted files to {outdir}")
            Path(outdir).mkdir(exist_ok=True)
            if outfile is not None:
                out = Path(outdir /  outfile.name)
            else:
                out = outdir.joinpath(Path(infile).name)
        elif outfile is not None:
            out = outfile
        print(f" Writing converted file to {Path(out).absolute}")

        writeQueryWithFullParagraphs(file_path=out, queryWithFullParagraphList=converted)



def main():
    """Entry point for the module."""
    # x = parseQueryWithFullParagraphs("./benchmarkY3test-qrels-with-text.jsonl.gz")
    # print(x[0])
    import argparse
    parser = argparse.ArgumentParser(description="Merge *jsonl.gz files")

    subparsers = parser.add_subparsers(dest='command', help="Choose one of the sub-commands")



    merge_parser = subparsers.add_parser('merge', help="Merge full paragraphs (xxx.jsonl.gz files) with or without grades into a single new file.")
    merge_parser.add_argument(dest='paragraph_file', type=Path, metavar='xxx.jsonl.gz', nargs='+'
                        , help='one or more json files with paragraph with or without exam grades.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )
    merge_parser.add_argument('-o','--out', type=str, metavar='FILE'
                        , help=f'output file that merged all input files'
                        )
        
    convert_parser = subparsers.add_parser('convert', help="change entries in full paragraphs files (xxx.jsonl.gz)")
    convert_parser.add_argument(dest='paragraph_file', type=str,metavar='xxx.jsonl.gz', nargs='+'
                        , help='one or more json files with paragraph with or without exam grades.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )
    convert_parser.add_argument('--ranking-method', type=str, metavar="NAME", help="Change entry to paragraph_data.rankings[].method to NAME")
    convert_parser.add_argument('--grade-llm', type=str, metavar="NAME", help="Change entry to exam_grades[].llm to NAME")
    convert_parser.add_argument('--grading-prompt', type=str, metavar="NAME", help="Change entry to exam_grades[].llm_info[prompt_class] to NAME, but only when it was previously set to --old-grading-prompt)")
    convert_parser.add_argument('--old-grading-prompt', type=str, metavar="NAME", help="Old value for --grading-prompt.  Can be set to None, to fix legacy configuations.")
    convert_parser.add_argument('-d','--out-dir', type=Path, metavar='DIR'
                        , help=f'output directory that converted files will be written to, using the same basename'
                        )
    convert_parser.add_argument('-o','--out-file', type=Path, metavar='FILE'
                        , help=f'output directory that converted file will be written to (only applies when only a single input file is given)'
                        )

    args = parser.parse_args()

    if args.command == "merge":
        merge(files=args.paragraph_file, out=args.out)

    elif args.command == "convert":
        convert(files=args.paragraph_file, outdir=args.out_dir, outfile=args.out_file, ranking_method=args.ranking_method, grade_llm=args.grade_llm, old_grading_prompt=args.old_grading_prompt, grading_prompt=args.grading_prompt)

if __name__ == "__main__":
    main()

