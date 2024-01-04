

from collections import defaultdict
from math import nan
import math
import statistics
from question_types import *
from parse_qrels_runs_with_text import *
from typing import Set, List, Tuple
from collections import defaultdict

from pydantic import BaseModel
import gzip

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
                                    for grade in para.exam_grades_iterable()
                                    ])
    return len(answered)


def totalCorrectQuestions(paras:List[FullParagraphData])->int:
    correct:Set[str] = set().union(*[set(grade.correctAnswered) 
                                for para in paras 
                                    for grade in para.exam_grades_iterable()
                                    ])
    return len(correct)

def examCoverageScore(paras:List[FullParagraphData], total_questions:Optional[int]=None)->float:
    '''Compute the exam coverage score for one query (and one method), based on a list of exam-graded `FullParagraphData`'''
    num_correct =totalCorrectQuestions(paras)
    if total_questions is None:
        total_questions = totalQuestions(paras)
    examCoverScore = frac(num_correct, total_questions)
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



OVERALL_ENTRY = "_overall_"

def compute_exam_cover_scores(query_paragraphs:List[QueryWithFullParagraphList], rank_cut_off:int=20)-> ExamCoverEvalsDict[str, ExamCoverEvals] :
    '''Workhorse to compute exam cover scores from exam-annotated paragraphs.
    Load input file with `parseQueryWithFullParagraphs`
    Write output file with `write_exam_results`
    or use convenience function `compute_exam_cover_scores_file`
    '''
    resultsPerMethod:ExamCoverEvalsDict[str, ExamCoverEvals] = ExamCoverEvalsDict()
    
    for queryWithFullParagraphList in query_paragraphs:

        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs


        total_correct = totalCorrectQuestions(paragraphs)
        total_questions = totalQuestions(paragraphs)
        overallExam = examCoverageScore(paragraphs)

        print(f'{query_id}, overall ratio {overallExam}')
        resultsPerMethod[OVERALL_ENTRY].examCoverPerQuery[query_id]=overallExam
        resultsPerMethod[OVERALL_ENTRY].nExamCoverPerQuery[query_id]=1.0

        # collect top paragraphs per method
        top_per_method = top_ranked_paragraphs(rank_cut_off, paragraphs)

        # computer query-wise exam scores for all methods
        for method, paragraphs in top_per_method.items():
            nexamScore = examCoverageScore(paragraphs, total_questions = total_correct)
            resultsPerMethod[method].nExamCoverPerQuery[query_id] = nexamScore

            examScore = examCoverageScore(paragraphs, total_questions = total_questions)
            resultsPerMethod[method].examCoverPerQuery[query_id] = examScore


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

    # aggregate query-wise exam scores into overall scores.
    for  method,examEvals in resultsPerMethod.items():
        examEvals.nExamScore, examEvals.nExamScoreStd = overallExam(examEvals.nExamCoverPerQuery.values())
        # print(f'OVERALL N-EXAM@{rank_cut_off} method {method}: avg examScores {avgExam:.2f} +/0 {stdExam:.3f}')

        examEvals.examScore, examEvals.examScoreStd = overallExam(examEvals.examCoverPerQuery.values())
        # print(f'OVERALL EXAM@{rank_cut_off} method {method}: avg examScores {avgExam:.2f} +/0 {stdExam:.3f}')

    return resultsPerMethod

def compute_exam_cover_scores_file(exam_input_file:Path, out_jsonl_file:Path, rank_cut_off:int=20):
    """which method covers most questions? """
    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)
    
    resultsPerMethod = compute_exam_cover_scores(query_paragraphs, rank_cut_off)
    
    write_exam_results(out_jsonl_file, resultsPerMethod)

def write_exam_results(out_jsonl_file, resultsPerMethod):
    with gzip.open(out_jsonl_file, 'wt', encoding='utf-8') as file:
        file.writelines([results.json()+'\n' for results in resultsPerMethod.values()])
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

    # Parse the arguments
    args = parser.parse_args()    
    compute_exam_cover_scores_file(exam_input_file=args.exam_annotated_file, out_jsonl_file=args.output)


if __name__ == "__main__":
    main()
