
import abc
from collections import defaultdict
from math import nan
import math
import statistics
from pydantic import BaseModel
import gzip
from typing import Set, List, Tuple, Dict, Optional, Iterable
from collections import defaultdict
from pathlib import Path

from .question_types import *
from .parse_qrels_runs_with_text import *
from .parse_qrels_runs_with_text import GradeFilter
from .pydantic_helper import pydantic_dump




def frac(num:int,den:int)->float:
    return ((1.0 * num) / (1.0 * den)) if den>0 else 0.0




class ExamCoverScorer(abc.ABC):
    '''Computes per-query EXAM and n-EXAM scores. Please construct via ExamCoverScoreFactory'''
    
    def __init__(self):
        self.totalQuestions:int = 0
        self.totalCorrect:int = 0

    @abc.abstractmethod
    def _examCoverageScore(self, paras:List[FullParagraphData], normalizer:int)->float:
        pass


    def plainExamCoverageScore(self, method_paras:List[FullParagraphData])->float:
        '''Plain EXAM cover score: fraction of all questions that could be correctly answered with the provided `method_paras`'''
        return self._examCoverageScore(paras=method_paras,  normalizer=self.totalQuestions)

    def nExamCoverageScore(self, method_paras:List[FullParagraphData])-> float:
        '''Normalized EXAM cover score: fraction of all questions that could be correctly answered with the provided `method_paras`, normalized by the set of questions that were answerable with any available text (as given in `all_paras`)'''
        return self._examCoverageScore(paras=method_paras, normalizer=self.totalCorrect)


class ExamCoverScorerCorrectAnswer(ExamCoverScorer):
    def __init__(self, grade_filter:GradeFilter
                 , paragraphs_for_normalization:Optional[List[FullParagraphData]]=None
                 , totalCorrect: Optional[int]=None
                 , totalQuestions:Optional[int]=None):
        super().__init__()
        self.grade_filter = grade_filter

        if totalQuestions is not None:
            self.totalQuestions = totalQuestions
        elif(paragraphs_for_normalization is not None):
            self.totalQuestions = self.__countTotalQuestions(paras=paragraphs_for_normalization)
        else:
            raise RuntimeError("Must set either `totalQuestions` or `paragraphs_for_normalization`")
        

        if totalCorrect is not None:
            self.totalCorrect = totalCorrect
        elif(paragraphs_for_normalization is not None):
            self.totalCorrect = self.__countTotalCorrectQuestions(paras=paragraphs_for_normalization)
        else:
            raise RuntimeError("Must set either `totalCorrect` or `paragraphs_for_normalization`")

        

    def __countTotalCorrectQuestions(self,paras:List[FullParagraphData])->int:
        correct:Set[str] = set().union(*[set(grade.correctAnswered) 
                                    for para in paras 
                                        for grade in para.retrieve_exam_grade(grade_filter=self.grade_filter)
                                        ])
        return len(correct)

    def __countTotalQuestions(self,paras:List[FullParagraphData])->int:
        answered:Set[str] = set().union(*[set(grade.correctAnswered + grade.wrongAnswered) 
                                    for para in paras 
                                        for grade in para.retrieve_exam_grade(grade_filter=self.grade_filter)
                                        ])
        return len(answered)


    def _examCoverageScore(self, paras:List[FullParagraphData], normalizer:int)->float:
        '''Compute the exam coverage score for one query (and one method), based on a list of exam-graded `FullParagraphData`'''
        num_correct = self.__countTotalCorrectQuestions(paras)
        exam_score = frac(num_correct, normalizer)
        return exam_score



class ExamCoverScorerSelfRatings(ExamCoverScorer):

    def __init__(self, grade_filter:GradeFilter
                 , min_self_rating:int
                 , paragraphs_for_normalization:Optional[List[FullParagraphData]]=None
                 , totalCorrect: Optional[int]=None
                 , totalQuestions:Optional[int]=None):
        super().__init__()
        self.grade_filter = grade_filter
        self.min_self_rating = min_self_rating

        if totalQuestions is not None:
            self.totalQuestions = totalQuestions
        elif(paragraphs_for_normalization is not None):
            self.totalQuestions = self.__countTotalQuestions(paras=paragraphs_for_normalization)
        else:
            raise RuntimeError("Must set either `totalQuestions` or `paragraphs_for_normalization`")
        

        if totalCorrect is not None:
            self.totalCorrect = totalCorrect
        elif(paragraphs_for_normalization is not None):
            self.totalCorrect = self.__countTotalCorrectQuestions(paras=paragraphs_for_normalization)
        else:
            raise RuntimeError("Must set either `totalCorrect` or `paragraphs_for_normalization`")


    def __countTotalCorrectQuestions(self,paras:List[FullParagraphData])->int:
        correct:Set[str] = { rate.question_id
                                            for para in paras 
                                                for grade in para.retrieve_exam_grade(grade_filter=self.grade_filter)
                                                        for rate in grade.self_ratings_as_iterable()
                                                            if rate.self_rating >= self.min_self_rating
                            }
        return len(correct)

    def __countTotalQuestions(self,paras:List[FullParagraphData])->int:
        answered:Set[str] = { rate.question_id
                                            for para in paras 
                                                for grade in para.retrieve_exam_grade(grade_filter=self.grade_filter)
                                                        for rate in grade.self_ratings_as_iterable()
                            }
        return len(answered)

    def _examCoverageScore(self, paras:List[FullParagraphData], normalizer:int)->float:
        '''Compute the exam coverage score for one query (and one method), based on a list of exam-graded `FullParagraphData`'''
        num_correct = self.__countTotalCorrectQuestions(paras=paras)
        exam_score = frac(num_correct, normalizer)
        return exam_score




# ---------------------------------
class ExamCoverScorerFactory():
    '''Factory for initializing ExamCoverScorers'''

    def __init__(self, grade_filter:GradeFilter, min_self_rating:Optional[int]):
        ''' if min_self_rating == None
                will use `grade.correctAnswer`
            if min_self_rating is set,
                will use `grade.self_ratings` with the respective minimum rate.
        '''
        self.grade_filter = grade_filter
        self.min_self_rating = min_self_rating


    def produce_from_paragraphs(self, paragraphs_for_normalization:Optional[List[FullParagraphData]]) -> ExamCoverScorer:
        if self.min_self_rating is None:
            return ExamCoverScorerCorrectAnswer(grade_filter=self.grade_filter
                                                , paragraphs_for_normalization=paragraphs_for_normalization
                                                )
        else:
            return ExamCoverScorerSelfRatings( grade_filter=self.grade_filter 
                                              , min_self_rating=self.min_self_rating
                                              , paragraphs_for_normalization=paragraphs_for_normalization
                                              )

    def produce_from_counts(self,  totalCorrect: int, totalQuestions:int)->ExamCoverScorer:
        if self.min_self_rating is None:
            return ExamCoverScorerCorrectAnswer(grade_filter=self.grade_filter
                                                , totalCorrect=totalCorrect
                                                , totalQuestions = totalQuestions
                                                )
        else:
            return ExamCoverScorerSelfRatings(  grade_filter=self.grade_filter
                                                , min_self_rating=self.min_self_rating
                                                , totalCorrect=totalCorrect
                                                , totalQuestions = totalQuestions
                                                )
                                            
    
# -------------------------------

# @dataclass
class ExamCoverEvals(BaseModel):
   method:str
   examCoverPerQuery: Dict[str,float]
   nExamCoverPerQuery: Dict[str,float]
   examScore: float
   nExamScore: float
   examScoreStd: float
   nExamScoreStd: float

class ExamCoverEvalsDict(dict):
    def __missing__(self, key:str)->ExamCoverEvals:
        value = ExamCoverEvals(method=key, examCoverPerQuery=dict(), nExamCoverPerQuery=dict(), examScore=nan, nExamScore=nan, examScoreStd=nan, nExamScoreStd=nan)
        self[key] = value
        return value



OVERALL_ENTRY = "_overall_"

def compute_exam_cover_scores(query_paragraphs:List[QueryWithFullParagraphList], exam_factory: ExamCoverScorerFactory, rank_cut_off:int=20)-> Dict[str, ExamCoverEvals] :
    '''Workhorse to compute exam cover scores from exam-annotated paragraphs.
    Load input file with `parseQueryWithFullParagraphs`
    Write output file with `write_exam_results`
    or use convenience function `compute_exam_cover_scores_file`
    '''
    resultsPerMethod:ExamCoverEvalsDict = ExamCoverEvalsDict()
    
    for queryWithFullParagraphList in query_paragraphs:

        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs

        # exam_cover_scorer = ExamCoverScorer(grade_filter=grade_filter, paragraphs_for_normalization=paragraphs)
        exam_cover_scorer = exam_factory.produce_from_paragraphs(paragraphs_for_normalization=paragraphs)

        overallExamScore = exam_cover_scorer.plainExamCoverageScore(paragraphs)

        print(f'{query_id}, overall ratio {overallExamScore}')
        resultsPerMethod[OVERALL_ENTRY].examCoverPerQuery[query_id]=overallExamScore
        resultsPerMethod[OVERALL_ENTRY].nExamCoverPerQuery[query_id]=1.0

        # collect top paragraphs per method
        top_per_method = top_ranked_paragraphs(rank_cut_off, paragraphs)

        # computer query-wise exam scores for all methods
        for method, paragraphs in top_per_method.items():
            nexamScore = exam_cover_scorer.nExamCoverageScore(paragraphs)
            resultsPerMethod[method].nExamCoverPerQuery[query_id] = nexamScore

            examScore = exam_cover_scorer.plainExamCoverageScore(paragraphs)
            resultsPerMethod[method].examCoverPerQuery[query_id] = examScore



    # aggregate query-wise exam scores into overall scores.

    def overallExam(examCoverPerQuery:List[float])->Tuple[float,float]:
        if(len(examCoverPerQuery)>=1):
            avgExam =   statistics.mean(examCoverPerQuery)
            stdExam = 0.0
            if(len(examCoverPerQuery)>=2):
                stdDevExam =   statistics.stdev(examCoverPerQuery) if(len(examCoverPerQuery)>=2) else 0.0
                stdExam = stdDevExam / math.sqrt(len(examCoverPerQuery))
            return (avgExam, stdExam)
        else:
            return (0.0,0.0)

    for  method,examEvals in resultsPerMethod.items():
        examEvals.nExamScore, examEvals.nExamScoreStd = overallExam(examEvals.nExamCoverPerQuery.values())
        print(f'OVERALL N-EXAM@{rank_cut_off} method {method}: avg examScores {examEvals.nExamScore:.2f} +/0 { examEvals.nExamScoreStd:.3f}')

        examEvals.examScore, examEvals.examScoreStd = overallExam(examEvals.examCoverPerQuery.values())
        print(f'OVERALL EXAM@{rank_cut_off} method {method}: avg examScores {examEvals.examScore:.2f} +/0 {examEvals.examScoreStd:.3f}')

    return resultsPerMethod

def compute_exam_cover_scores_file(exam_input_file:Path, out_jsonl_file:Path, exam_factory:ExamCoverScorerFactory, rank_cut_off:int=20):
    """export ExamCoverScores to a file:  which method covers most questions? """
    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)
    
    resultsPerMethod = compute_exam_cover_scores(query_paragraphs, exam_factory=exam_factory, rank_cut_off=rank_cut_off)
    
    write_exam_results(out_jsonl_file, resultsPerMethod)

def write_exam_results(out_jsonl_file, resultsPerMethod):
    with gzip.open(out_jsonl_file, 'wt', encoding='utf-8') as file:
        file.writelines([pydantic_dump(results)+'\n' for results in resultsPerMethod.values()])
        file.close()

def top_ranked_paragraphs(rank_cut_off:int, paragraphs:List[FullParagraphData])-> Dict[str,List[FullParagraphData]] :
    top_per_method:Dict[str,List[FullParagraphData]] = defaultdict(list)
    for para in paragraphs:
            for rank in para.paragraph_data.rankings:
                if rank.rank <= rank_cut_off:
                    top_per_method[rank.method].append(para)
    return top_per_method



def main():
    import argparse

    desc = f'''Compute the  Exam-Cover evaluation scores from ranked paragraphs with exam annotations. \n
               \n
              The input file (i.e, exam_annotated_file) has to be a *JSONL.GZ file that follows this structure: \n
              \n  
                  [query_id, [FullParagraphData]] \n
              \n
               where `FullParagraphData` meets the following structure \n
             {FullParagraphData.schema_json(indent=2)} \n
              \n 
             The output format is a jsonl.gz file, containing one line for each method's exam result computation. The format is as follows: \n
             {ExamCoverEvals.schema_json(indent=2)}\n
             '''
    

    # Create the parser
    # parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser = argparse.ArgumentParser(description="Compute the  Exam-Cover evaluation scores from ranked paragraphs with exam annotations."
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exam_annotated_file', type=str, metavar='exam-xxx.jsonl.gz'
                        , help='json file that annotates each paragraph with a number of anserable questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )

    # Add an optional output file argument
    parser.add_argument('-o', '--output', type=str, metavar="FILE", help='Output JSONL.gz file name, where exam results will be written to', default='output.qrels')

    parser.add_argument('-m', '--model', type=str, metavar="MODEL_NAME", help='name of huggingface model that created exam grades')
    parser.add_argument('--prompt-class', type=str, choices=get_prompt_classes(), required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))
    parser.add_argument('-r', '--use-ratings', action='store_true', help='If set, correlation analysis will use graded self-ratings. Default is to use the number of correct answers.')
    parser.add_argument('--question-set', type=str, choices=["tqa","naghmeh","question-bank"], metavar="SET ", help='Which question set to use. Options: tqa or naghmeh ')

    # Parse the arguments
    args = parser.parse_args()    
    grade_filter = GradeFilter(model_name=args.model, prompt_class = args.prompt_class, is_self_rated=None, min_self_rating=None, question_set=args.question_set)


    # Parse the arguments
    args = parser.parse_args()    
    exam_factory = ExamCoverScorerFactory(grade_filter=grade_filter)

    compute_exam_cover_scores_file(exam_input_file=args.exam_annotated_file, out_jsonl_file=args.output, model_name=args.model, exam_factory=exam_factory)


if __name__ == "__main__":
    main()
