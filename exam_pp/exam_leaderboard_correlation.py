from collections import defaultdict
import statistics
from question_types import *
from parse_qrels_runs_with_text import *
from typing import Set, List, Tuple

from exam_cover_metric import *


manualLeaderboard = { "dangnt-nlp": 1
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

origExamLeaderboard = { "ReRnak2_BERT": 1
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
class CorrelationStats:
    spearman_correlation:float
    kendall_correlation:float

def leaderboard_rank_correlation(systemEval:Dict[str,float])->CorrelationStats:
    from scipy.stats import spearmanr, kendalltau, rankdata

    methods = list(manualLeaderboard.keys())
    ranks1 = [manualLeaderboard[method] for method in methods]

    # Extract scores for the methods
    scores = [systemEval[method] for method in methods]
    # Use rankdata to handle ties in scoring
    ranks2 = rankdata([-score for score in scores], method='average')  # Negative scores for descending order

    
    # Calculate Spearman's Rank Correlation Coefficient
    spearman_correlation, spearman_p_value = spearmanr(ranks1, ranks2)
    kendall_res = kendalltau(ranks1, ranks2)

    return CorrelationStats(spearman_correlation=spearman_correlation, kendall_correlation=kendall_res.correlation)

@dataclass
class LeaderboardEntry:
    method:str
    eval_score:float
    rank:int

def create_leaderboard(systemEval:Dict[str,float])->List[LeaderboardEntry]:
    systemEvalSorted = sorted(list(systemEval.items()), key=lambda x: x[1], reverse=True)
    systemEvalRanked = [LeaderboardEntry(method, score, rank) for rank, (method, score) in enumerate(systemEvalSorted, start=1)]
    return systemEvalRanked


def print_leaderboard_eval_file(exam_result_file:Path):
    def read_result_file()->[ExamCoverEvals]:
        # Open the gzipped file
        with gzip.open(exam_result_file, 'rt', encoding='utf-8') as file:
            return [ExamCoverEvals.parse_raw(line) for line in file]

    evals = read_result_file()

    print_leaderboard_eval(evals)
    pass

def leaderboard_table(evals:List[ExamCoverEvals]):
    evals_ = sorted(evals, key= lambda eval: eval.nExamScore, reverse=True)

    def f2s(x:float)->str:
        return f'{x:.3f}'
    def i2s(x:int)->str:
        return f'{x}'
    
    header = '\t'.join(['method'
                        ,'exam','+/-','exam-std'
                        ,'n-exam','+/-','n-exam-std'
                        ,'orig_TREC_leaderboard_rank'
                        ,'orig_EXAM_leaderboard_rank'
                        ])

    lines = [ '\t'.join([e.method
                        ,f2s(e.examScore), '+/-', f2s(e.examScoreStd)
                        ,f2s(e.nExamScore), '+/-', f2s(e.nExamScoreStd)
                        , i2s(manualLeaderboard.get(e.method,' '))
                        , i2s(origExamLeaderboard.get(e.method,' '))
                        ])
                             for e in evals_]
    print('\n'.join([header]+lines))

def print_leaderboard_eval(evals:List[ExamCoverEvals]):
    '''Print the Leaderboard in trec_eval evaluation output format.
    Load necessary data with `read_exam_result_file()` or use the convenience method `print_leaderboard_eval_file`
    '''
    nExamEval = {eval.method: eval.nExamScore for eval in evals}
    examEval = {eval.method: eval.examScore for eval in evals}

    print("\n".join( ["\t".join([x.method, "exam", f'{x.eval_score:.4}']) 
                        for x in create_leaderboard(examEval)]))
    print("\n".join( ["\t".join([x.method, "n-exam", f'{x.eval_score:.4}' ]) 
                        for x in create_leaderboard(nExamEval)]))


def read_exam_result_file(exam_result_file:Path)->List[ExamCoverEvals]:
    # Open the gzipped file
    with gzip.open(exam_result_file, 'rt', encoding='utf-8') as file:
        return [ExamCoverEvals.parse_raw(line) for line in file]

def leaderboard_correlation_files(exam_result_file:Path):
    evals = read_exam_result_file(exam_result_file)

    nExamCorrelation, examCorrelation = leaderboard_correlation(evals)
    print(f' nExam:{nExamCorrelation}')
    print(f' exam:{examCorrelation}')

def leaderboard_correlation(evals:List[ExamCoverEvals])->Tuple[CorrelationStats,CorrelationStats]:
    '''Compute Leaderboard correlation. 
    Load necessary data with `read_exam_result_file()` or use the convenience method `leaderboard_correlation_files`
    '''
    nExamEval = {eval.method: eval.nExamScore for eval in evals}
    examEval = {eval.method: eval.examScore for eval in evals}

    nExamCorrelation = leaderboard_rank_correlation(nExamEval)
    examCorrelation = leaderboard_rank_correlation(examEval)
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

    args = parser.parse_args()  

    if args.eval:
        print_leaderboard_eval_file(exam_result_file=args.exam_result_file)
    if args.rank_correlation:
        leaderboard_correlation_files(exam_result_file=args.exam_result_file)


if __name__ == "__main__":
    main()
