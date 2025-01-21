from collections import defaultdict
from math import sqrt
from pathlib import Path
import re
# from statistics import mean, stdev
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple


from .exam_cover_metric import OVERALL_ENTRY, ExamCoverEvals, ExamCoverEvalsDict, systemMeanStderr

def run_trec_eval_variance(run_dir:Path, qrels:Path, min_level:Optional[int], trec_eval_metric:str, trec_eval_args:str="-n -c")->str:
    # Define the command to be executed
    l_arg = f" -l {min_level} " if min_level is not None else ""
    command = f"for f in *.run; do  res=`trec_eval -q {trec_eval_args} -m {trec_eval_metric} {l_arg} {qrels.resolve().as_posix()} $f`; echo \"$f $res\"; done"
    print(f'Running trec_eval command:\n{command}\nin directory: {run_dir}')

    # Run the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=run_dir.resolve().as_posix())

    if result.stderr:
        print(f"Received command errors: "+result.stderr.strip())
        if result.stderr.startswith ('trec_eval: No queries with both results and relevance info'):
            raise RuntimeError(f'''trec_eval:  Queries in qrels file are different from queries in the run file.\n
                qrels file: {qrels}\n
                run files: {run_dir}/*run
                    ''')
    # Extract stdout
    output = result.stdout.strip()

    # print(f"Command output: {output}")
    return output


def parse_trec_eval_per_query(command_output:str)-> Dict[str,List[Tuple[str,float]]]:
    # eval_parse_pattern = r"^(\S+)\.run.*\s([0-9.]+)$"
    # eval_parse_pattern = r"^(\S+)\.run.*?\s+\S+\s+(\S+)\s+([0-9.]+)$"
    eval_parse_pattern = r"^(?:(\S+)\.run)?\s*\S+\s+(\S+)\s+([0-9.]+)$"

    # Explanation:
    #     (?: ... )?: A non-capturing group that is made optional by the ?. This means the pattern inside the group may or may not appear in the target string.
    #     (\S+)\.run: This is the part that captures the method name followed by .run. The entire group is now optional, meaning your expression can match lines where this pattern doesn’t exist.
    #     \s*: Zero or more whitespaces to handle any number of spaces that might appear when the method.run part is missing.
    #     \S+: Captures the metric identifier which we assume always follows any space after the optional method.run.
    #     \s+: One or more whitespace characters.
    #     (\S+): Captures the next significant word, e.g., "all".
    #     \s+: One or more whitespace characters before the score.
    #     ([0-9.]+): Captures the floating-point number representing the score.

    def parse_line(line:str, prev_method)->Tuple[str,str, float, str]:
        match = re.match(eval_parse_pattern, line)

        if match:
            method = match.group(1)
            if not method or not len(method):
                method=prev_method
            else:
                prev_method=method
                            
            query = match.group(2)
            score = float(match.group(3))
            # print(f"{method}\t{query}\t{score}")
            return (method, query, score, prev_method)
        else:
            raise RuntimeError(f"Can't parse trec_eval output. Offending line: \"{line}\".\nFull command output:\n{command_output}")

    method_per_query_results:Dict[str,List[Tuple[str,float]]] = defaultdict(list)
    prev_method=""
    for line in command_output.split("\n"):
        if len(line.strip())>0:
            (method, query, score, prev_method) = parse_line(line.strip(), prev_method)
            method_per_query_results[method].append((query,score))

    return method_per_query_results

def compute_exam_qrels_scores(method_per_query_results:Dict[str,List[Tuple[str,float]]])-> Dict[str, ExamCoverEvals] :
    '''Workhorse to compute exam qrels scores from trec_eval outputs
    # Write output file with `write_exam_results`
    # or use convenience function `compute_exam_cover_scores_file`
    '''
    resultsPerMethod:ExamCoverEvalsDict = ExamCoverEvalsDict()

    resultsPerMethod[OVERALL_ENTRY].examScore=1.0
    resultsPerMethod[OVERALL_ENTRY].nExamScore=0.0

    for method, per_query_results in method_per_query_results.items():

        # compute query-wise exam scores for all methods
        for query_id,score in per_query_results:
            if query_id != "all":
                resultsPerMethod[method].examCoverPerQuery[query_id]=score


    for  method,examEvals in resultsPerMethod.items():
        examEvals.examScore, examEvals.examScoreStd = systemMeanStderr(examEvals.examCoverPerQuery.values())
        print(f'OVERALL EXAM-qrels method {method}: avg examScores {examEvals.examScore:.2f} +/0 {examEvals.examScoreStd:.3f}')


    return resultsPerMethod



def run_trec_eval(run_dir:Path, qrels:Path, min_level:Optional[int], trec_eval_metric:str):
    # Define the command to be executed
    l_arg = f" -l {min_level} " if min_level is not None else ""
    command = f"for f in *.run; do  res=`trec_eval -m {trec_eval_metric} {l_arg} {qrels.resolve().as_posix()} $f`; echo \"$f $res\"; done"
    print(f'Running trec_eval command:\n{command}\nin directory: {run_dir}')

    # Run the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=run_dir.resolve().as_posix())

    if result.stderr:
        print(f"Received command errors: "+result.stderr.strip())
        if result.stderr.startswith ('trec_eval: No queries with both results and relevance info'):
            raise RuntimeError(f'''trec_eval:  Queries in qrels file are different from queries in the run file.\n
                qrels file: {qrels}\n
                run files: {run_dir}/*run
                    ''')
    # Extract stdout
    output = result.stdout.strip()

    # print(f"Command output: {output}")
    return output

def mimic_trec_eval():
    output='''
Bert-ConvKNRM-50.run P_20                       all     0.4129
Bert-ConvKNRM.run P_20                          all     0.2512
Bert-DRMMTKS.run P_20                   all     0.1404
ECNU_BM25.run P_20                      all     0.4294
ECNU_BM25_1.run P_20                    all     0.4275
ECNU_ReRank1.run P_20                   all     0.3570
ICT-BM25.run P_20                       all     0.4182
ICT-DRMMTKS.run P_20                    all     0.1422
IRIT1.run P_20                          all     0.3876
IRIT2.run P_20                          all     0.3876
IRIT3.run P_20                          all     0.3876
ReRnak2_BERT.run P_20                   all     0.4499
ReRnak3_BERT.run P_20                   all     0.4476
UNH-bm25-ecmpsg.run P_20                        all     0.3410
UNH-bm25-rm.run P_20                    all     0.3853
UNH-bm25-stem.run P_20                          all     0.3410
UNH-dl100.run P_20                      all     0.3410
UNH-dl300.run P_20                      all     0.3410
UNH-ecn.run P_20                        all     0.2745
UNH-qee.run P_20                        all     0.4135
UNH-tfidf-lem.run P_20                          all     0.3410
UNH-tfidf-ptsim.run P_20                        all     0.3410
UNH-tfidf-stem.run P_20                         all     0.3410
UvABM25RM3.run P_20                     all     0.1916
UvABottomUp1.run P_20                   all     0.1327
UvABottomUp2.run P_20                   all     0.1845
UvABottomUpChangeOrder.run P_20                         all     0.1794
bm25-populated.run P_20                         all     0.3585
dangnt-nlp.run P_20                     all     0.4229
'''
    return output.strip()

def parse_trec_eval(command_output:str)->Dict[str,float]:
    eval_parse_pattern = r"^(\S+)\.run.*\s([0-9.]+)$"

    def parse_line(line:str)->Tuple[str,float]:
        match = re.match(eval_parse_pattern, line)

        if match:
            method = match.group(1)
            score = float(match.group(2))
            # print(f"{method}\t{score}")
            return (method, score)
        else:
            raise RuntimeError(f"Can't parse trec_eval output. Offending line: \"{line}\".\nFull command output:\n{command_output}")

    return dict([parse_line(line.strip()) 
                    for line in command_output.split("\n") 
                    if len(line.strip())>0
                ])

def trec_eval_leaderboard(run_dir:Path, qrels:Path, min_level:Optional[int],trec_eval_metric:str)-> Dict[str,float]:
    '''Designed to interoperate with `leaderboard_rank_correlation` '''
    output=run_trec_eval(run_dir=run_dir, qrels=qrels, min_level=min_level, trec_eval_metric=trec_eval_metric)
    # output=mimic_trec_eval()
    return parse_trec_eval(output)

def trec_eval_leaderboard_per_query(query_id:Optional[str], run_dir:Path, qrels:Path, min_level:Optional[int],trec_eval_metric:str)-> Dict[str,float]:
    '''Designed to interoperate with `leaderboard_rank_correlation` '''
    # per_query_qrels = Path("./tmp")/Path(f"{query_id}-{qrels.name}")

    query_id_str = query_id if query_id is not None else "ALL"
    fd, per_query_qrels_str = tempfile.mkstemp(suffix=".qrels", prefix=f"{query_id_str}-{qrels.name}-", text=True)
    per_query_qrels = Path(per_query_qrels_str)

    with open(per_query_qrels, mode="tw") as w:
        found = 0
        with open(qrels, mode="tr") as f:
            for line in f.readlines():
                if query_id is None or line.startswith(query_id):
                    w.write(line)
                    found += 1
    print(f"wrote {found} lines for query {query_id_str} to {per_query_qrels} from { qrels}")
    

    if found == 0:
        raise RuntimeError(f"No entries for query {query_id_str} found in qrels file {qrels}")

    output=run_trec_eval(run_dir=run_dir, qrels=per_query_qrels, min_level=min_level, trec_eval_metric=trec_eval_metric)
    per_query_qrels.unlink()
    return parse_trec_eval(output)