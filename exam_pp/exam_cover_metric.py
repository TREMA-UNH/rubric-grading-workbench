

from collections import defaultdict
from math import nan
import statistics
from question_types import *
from parse_qrels_runs_with_text import *
from typing import Set, List, Tuple

from pydantic import BaseModel
import gzip
import json

def plainExamCoverageScore(method_paras:List[FullParagraphData])->float:
    '''Plain EXAM cover score: fraction of all questions that could be correctly answered with the provided `method_paras`'''
    return examCoverageScore(paras=method_paras)

def nExamCoverageScore(method_paras:List[FullParagraphData], all_paras:List[FullParagraphData])-> float:
    '''Normalized EXAM cover score: fraction of all questions that could be correctly answered with the provided `method_paras`, normalized by the set of questions that were answerable with any available text (as given in `all_paras`)'''
    return examCoverageScore(paras=method_paras, total_questions=totalCorrectQuestions(paras=all_paras))



def frac(num:int,den:int)->float:
    return ((1.0 * num) / (1.0 * den)) if den>0 else 0.0


def totalQuestions(paras:List[FullParagraphData])->int:
    answered:Set[str] = set().union(*[set(grade.correctAnswered + grade.wrongAnswered) 
                                for para in paras 
                                    for grade in para.exam_grades
                                    ])
    return len(answered)


def totalCorrectQuestions(paras:List[FullParagraphData])->int:
    correct:Set[str] = set().union(*[set(grade.correctAnswered) 
                                for para in paras 
                                    for grade in para.exam_grades
                                    ])
    return len(correct)

def examCoverageScore(paras:List[FullParagraphData], total_questions:Optional[int]=None)->float:
    '''Compute the exam coverage score for one query (and one method), based on a list of exam-graded `FullParagraphData`'''
    correct:Set[str] = set().union(*[set(grade.correctAnswered) 
                                            for para in paras 
                                                for grade in para.exam_grades
                                                ])
    if total_questions is None:
        answered:Set[str] = set().union(*[set(grade.correctAnswered + grade.wrongAnswered) 
                                            for para in paras 
                                                for grade in para.exam_grades
                                                ])
        total_questions = len(answered)
    examCoverScore = frac(len(correct), total_questions)
    return examCoverScore


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


from collections import defaultdict


def computeExamCoverScores(exam_input_file:Path, out_jsonl_file:Path, rank_cut_off:int=20):
    """which method covers most questions? """
    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)

    examCoverPerMethod:Dict[str, List[float]] = defaultdict(list) 
    nexamCoverPerMethod:Dict[str, List[float]] = defaultdict(list) 

    resultsPerMethod:ExamCoverEvalsDict[str, ExamCoverEvals] = ExamCoverEvalsDict()

    
    for queryWithFullParagraphList in query_paragraphs:

        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs


        total_correct = totalCorrectQuestions(paragraphs)
        total_questions = totalQuestions(paragraphs)
        overallExam = examCoverageScore(paragraphs)
        examCoverPerMethod["_overall"].append(overallExam)
        nexamCoverPerMethod["_overall"].append(1.0)

        print(f'{query_id}, overall ratio {overallExam}')

        # collect top paragraphs per method
        top_per_method:Dict[str,List[FullParagraphData]] = defaultdict(list)
        for para in paragraphs:
                for rank in para.paragraph_data.rankings:
                    if rank.rank <= rank_cut_off:
                        top_per_method[rank.method].append(para)

        # computer query-wise exam scores for all methods
        for method, paragraphs in top_per_method.items():
            nexamScore = examCoverageScore(paragraphs, total_questions = total_correct)
            # nexamCoverPerMethod[method].append(nexamScore) #
            resultsPerMethod[method].nExamCoverPerQuery[query_id] = nexamScore

            examScore = examCoverageScore(paragraphs, total_questions = total_questions)
            # examCoverPerMethod[method].append(examScore) #
            resultsPerMethod[method].examCoverPerQuery[query_id] = examScore

    # aggregate query-wise exam scores into overall scores.
    for  method,examEvals in resultsPerMethod.items():
        if(len(examEvals.nExamCoverPerQuery)>=1):
            avgExam =   statistics.mean(examEvals.nExamCoverPerQuery.values())
            stdExam =   statistics.stdev(examEvals.nExamCoverPerQuery.values()) if(len(examEvals.nExamCoverPerQuery)>=2) else 0.0
            examEvals.nExamScore = avgExam
            examEvals.nExamScoreStd = stdExam
            # print(f'OVERALL N-EXAM@{rank_cut_off} method {method}: avg examScores {avgExam:.2f} +/0 {stdExam:.3f}')

        if(len(examEvals.examCoverPerQuery)>=1):
            avgExam =   statistics.mean(examEvals.examCoverPerQuery.values())
            stdExam =   statistics.stdev(examEvals.examCoverPerQuery.values()) if(len(examEvals.examCoverPerQuery)>=2) else 0.0
            examEvals.examScore = avgExam
            examEvals.examScoreStd = stdExam
            # print(f'OVERALL EXAM@{rank_cut_off} method {method}: avg examScores {avgExam:.2f} +/0 {stdExam:.3f}')

    # nExamEval:Dict[str,float] = {method: examEvals.nExamScore for method,examEvals in resultsPerMethod.items()}
    # examEval:Dict[str,float] = {method: examEvals.examScore for method,examEvals in resultsPerMethod.items()}    
    # print()

    # print(resultsPerMethod)
    # print()
    # print(examEval)
    # print("\n".join( [str(x)  for x in create_leaderboard(examEval)]))
    pass
    with gzip.open(out_jsonl_file, 'wt', encoding='utf-8') as file:
        file.writelines([results.json() for results in resultsPerMethod.values()])
        file.close()



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

    # Parse the arguments
    args = parser.parse_args()    
    computeExamCoverScores(exam_input_file=args.exam_annotated_file, out_jsonl_file=args.output)


if __name__ == "__main__":
    main()
