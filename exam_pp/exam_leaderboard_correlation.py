from collections import defaultdict
from typing import Set, List, Tuple, Dict, Union, Optional
from dataclasses import dataclass
from pathlib import Path

import statistics
import scipy
from scipy.stats import spearmanr, kendalltau, rankdata
import scipy.stats


from .question_types import *
from .parse_qrels_runs_with_text import *
from .parse_qrels_runs_with_text import GradeFilter
from .exam_cover_metric import *
from .exam_cover_metric import ExamCoverEvals
from . import exam_leaderboard_correlation



def load_leaderboard(file_path:Path)->Dict[str,float]:
    with open(file_path, 'r') as file:
        return json.load(file)

official_DL19_leaderboard:Dict[str,float] = {
                        "idst_bert_p1": 1,
                        "idst_bert_p2": 2,
                        "idst_bert_p3": 3,
                        "p_exp_rm3_bert": 4,
                        "p_bert": 5,
                        "ids_bert_pr2": 6,
                        "ids_bert_pr1": 7,
                        "p_exp_bert": 8,
                        "test1": 9,
                        "TUA1-1": 10,
                        "runid4": 11,
                        "runid3": 12,
                        "TUW19-p3-f": 13,
                        "TUW19-p1-f": 14,
                        "TUW19-p3-re": 15,
                        "TUW19-p1-re": 16,
                        "TUW19-p2-f": 17,
                        "ICT-BERT2": 18,
                        "srchvrs_ps_run2": 19,
                        "TUW19-p2-re": 20,
                        "ICT-CKNRM_B": 21,
                        "ms_duet_passage": 22,
                        "ICT-CKNRM_B50": 23,
                        "srchvrs_ps_run3": 24,
                        "bm25tuned_prf_p": 25,
                        "bm25base_ax_p": 26,
                        "bm25tuned_ax_p": 27,
                        "bm25base_prf_p": 28,
                        "runid2": 29,
                        "runid5": 30,
                        "bm25tuned_rm3_p": 31,
                        "bm25base_rm3_p": 32,
                        "bm25base_p": 33,
                        "srchvrs_ps_run1": 34,
                        "bm25tuned_p": 35,
                        "UNH_bm25": 36
                    }

official_CarY3_leaderboard:Dict[str,float] = { "dangnt-nlp": 1
                                            , "ReRnak3_BERT": 2
                                            , "ReRnak2_BERT": 3
                                            , "IRIT1" : 5
                                            , "IRIT2" : 5
                                            , "IRIT3" : 5
                                            , "ECNU_BM25_1" : 7.5
                                            , "ECNU_ReRank1" : 7.5
                                            , "Bert-ConvKNRM-50" : 9
                                            , "bm25-populated" : 10
                                            , "UNH-bm25-ecmpsg" : 11
                                            , "Bert-DRMMTKS" : 12
                                            , "UvABM25RM3" : 13
                                            , "UvABottomUpChangeOrder" : 14
                                            , "UvABottomUp2" : 15
                                            , "ICT-DRMMTKS" : 16
}

origExamLeaderboard:Dict[str,float]  = { "ReRnak2_BERT": 1
                      , "dangnt-nlp": 2
                      , "Bert-DRMMTKS": 3
                      , "IRIT2": 4
                      , "ReRnak3_BERT": 5
                      , "Bert-ConvKNRM-50": 6
                      , "IRIT1": 7
                      , "bm25-populated": 8
                      , "IRIT3": 9
                      , "UNH-bm25-ecmpsg": 10
                      , "ECNU_BM25_1": 11
                      , "ECNU_ReRank1": 12
                      , "ICT-DRMMTKS": 13
                      , "UvABottomUpChangeOrder": 14
                      , "UvABM25RM3": 15
                      , "UvABottomUp2": 16
}

@dataclass
class CorrelationStats():
    spearman_correlation:float
    kendall_correlation:float

    def pretty_print(self)->str:
        return f'spearman_correlation ={self.spearman_correlation:.2f}\tkendall_correlation = {self.kendall_correlation:.2f}'


def compatible_kendalltau(ranks1, ranks2)->Tuple[float,float]:
    from packaging import version

    if version.parse(scipy.__version__) >= version.parse('1.7.0'):    
    # if scipy.__version__ >= '1.7.0':
        # For scipy 1.7.0 and later
        tau, p_value = kendalltau(ranks1, ranks2)
        return tau, p_value
    else:
        # For older versions
        from scipy.stats import SignificanceResult
        result = kendalltau(ranks1, ranks2)
        return result.correlation, result.pvalue


def leaderboard_rank_correlation(systemEval:Dict[str,float], official_leaderboard:Dict[str,int])->CorrelationStats:
    methods = list(official_leaderboard.keys())
    ranks1 = [official_leaderboard[method] for method in methods]

    for method in methods:
        if not method in systemEval:
            raise RuntimeError(f'official leaderboard contains method {method}, but predicted leaderboard does not.  \nMethods in predicted leaderboard:{systemEval.keys()} \nMethods in official leaderboard {methods}')

    # Extract scores for the methods
    scores = [systemEval[method] for method in methods]
    # Use rankdata to handle ties in scoring
    ranks2 = rankdata([-score for score in scores], method='average')  # Negative scores for descending order

    
    # Calculate Spearman's Rank Correlation Coefficient
    spearman_correlation:float
    spearman_p_value:float
    kendall_correlation:float
    kendall_p_value:float
    spearman_correlation, spearman_p_value = spearmanr(ranks1, ranks2)
    kendall_correlation, kendall_p_value = compatible_kendalltau(ranks1, ranks2)

    return CorrelationStats(spearman_correlation=spearman_correlation, kendall_correlation=kendall_correlation)

@dataclass
class LeaderboardEntry:
    method:str
    eval_score:float
    rank:int

def create_leaderboard(systemEval:Dict[str,float])->List[LeaderboardEntry]:
    systemEvalSorted = sorted(list(systemEval.items()), key=lambda x: x[1], reverse=True)
    systemEvalRanked = [LeaderboardEntry(method=method, eval_score=score, rank=rank) for rank, (method, score) in enumerate(systemEvalSorted, start=1)]
    return systemEvalRanked


def print_leaderboard_eval_file(exam_result_file:Path, grade_filter:GradeFilter):
    import gzip
    def read_result_file()->List[ExamCoverEvals]:
        # Open the gzipped file
        with gzip.open(exam_result_file, 'rt', encoding='utf-8') as file:
            return [ExamCoverEvals.parse_raw(line) for line in file]

    evals = read_result_file()

    print_leaderboard_eval(evals, grade_filter=grade_filter)
    pass

def leaderboard_table(evals:List[ExamCoverEvals], official_leaderboard:Dict[str,int])->[str]:
    evals_ = sorted(evals, key= lambda eval: eval.nExamScore, reverse=True)

    def f2s(x:Optional[float])->str:
        if x is None:
            return ' '
        else:
            return f'{x:.3f}'
    # def i2s(x:Union[int,str])->str:
    #     return f'{x}'
    
    header = '\t'.join(['method'
                        ,'exam','+/-','exam-std'
                        ,'n-exam','+/-','n-exam-std'
                        ,'orig_TREC_leaderboard_rank'
                        ,'orig_EXAM_leaderboard_rank'
                        ])

    lines = [ '\t'.join([e.method
                        ,f2s(e.examScore), '+/-', f2s(e.examScoreStd)
                        ,f2s(e.nExamScore), '+/-', f2s(e.nExamScoreStd)
                        , f2s(official_leaderboard.get(e.method))
                        , f2s(origExamLeaderboard.get(e.method))
                        ])
                             for e in evals_]
    # print('\n'.join([f'EXAM scores produced with {grade_filter}', header]+lines))
    return [header]+lines


def print_leaderboard_eval(evals:List[ExamCoverEvals], grade_filter:GradeFilter):
    '''Print the Leaderboard in trec_eval evaluation output format.
    Load necessary data with `read_exam_result_file()` or use the convenience method `print_leaderboard_eval_file`
    '''
    nExamEval = {eval.method: eval.nExamScore for eval in evals}
    examEval = {eval.method: eval.examScore for eval in evals}

    print(f'EXAM scores produced with {grade_filter}')
    print("\n".join( ["\t".join([x.method, "exam", f'{x.eval_score:.4}']) 
                        for x in create_leaderboard(examEval)]))
    print("\n".join( ["\t".join([x.method, "n-exam", f'{x.eval_score:.4}' ]) 
                        for x in create_leaderboard(nExamEval)]))


def read_exam_result_file(exam_result_file:Path)->List[ExamCoverEvals]:
    import gzip
    # Open the gzipped file
    with gzip.open(exam_result_file, 'rt', encoding='utf-8') as file:
        return [ExamCoverEvals.parse_raw(line) for line in file]

def leaderboard_correlation_files(exam_result_file:Path, official_leaderboard:Dict[str,int]):
    evals = read_exam_result_file(exam_result_file)

    nExamCorrelation, examCorrelation = leaderboard_correlation(evals, official_leaderboard=official_leaderboard)
    print(f' nExam:{nExamCorrelation}')
    print(f' exam:{examCorrelation}')

def leaderboard_correlation(evals:Iterable[ExamCoverEvals], official_leaderboard:Dict[str,int])->Tuple[CorrelationStats,CorrelationStats]:
    '''Compute Leaderboard correlation. 
    Load necessary data with `read_exam_result_file()` or use the convenience method `leaderboard_correlation_files`
    '''
    nExamEval = {eval.method: eval.nExamScore for eval in evals}
    examEval = {eval.method: eval.examScore for eval in evals}

    nExamCorrelation = leaderboard_rank_correlation(nExamEval, official_leaderboard=official_leaderboard)
    examCorrelation = leaderboard_rank_correlation(examEval, official_leaderboard=official_leaderboard)
    return nExamCorrelation,examCorrelation




def main():
    import argparse

    desc = f'''Compute the  Exam-Cover evaluation scores from ranked paragraphs with exam annotations. \n
               \n
              The input file (i.e, exam_annotated_file) has to be a *JSONL.GZ file that follows this structure: \n
              \n  
                  ExamCoverEvals \n
              \n
               where `ExamCoverEvals` meets the following structure \n
             {ExamCoverEvals.schema_json(indent=2)} \n
             '''
    

    parser = argparse.ArgumentParser(description="Analyze leaderboard correlation of exam results for each system."
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exam_result_file', type=str, metavar='exam-result.jsonl.gz'
                        , help='json file with exam evaluation scores. The typical file pattern is `exam-xxx.jsonl.gz.'
                        )

    parser.add_argument('-l', '--leaderboard', action='store_true', help='Print leaderbord evaluation data')
    parser.add_argument('-c', '--rank-correlation', action='store_true', help='Print leaderbord rank correlation')

    # parser.add_argument('-o', '--output', type=str, metavar="FILE", help='Output JSONL.gz file name, where exam results will be written to', default='output.qrels')
    parser.add_argument('--testset', type=str, choices=["cary3","dl19"], metavar="SET ", help='Which question set to use. Options: tqa or naghmeh ')

    # Parse the arguments
    args = parser.parse_args(args)    
    
    official_leaderboard:Dict[str,int]
    if args.testset == "cary3":
        official_leaderboard = exam_leaderboard_correlation.official_CarY3_leaderboard 
    elif args.testset == "dl19":
        official_leaderboard = exam_leaderboard_correlation.official_DL19_leaderboard


    if args.eval:
        print_leaderboard_eval_file(exam_result_file=args.exam_result_file)
    if args.rank_correlation:
        leaderboard_correlation_files(exam_result_file=args.exam_result_file, official_leaderboard=official_leaderboard)


if __name__ == "__main__":
    main()
