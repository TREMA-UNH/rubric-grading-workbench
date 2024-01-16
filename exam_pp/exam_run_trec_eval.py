from pathlib import Path
import re
import subprocess
from typing import Dict, Optional, Tuple

def run_trec_eval(run_dir:Path, qrels:Path, min_level:Optional[int]):
    # Define the command to be executed
    l_arg = f" -l {min_level} " if min_level is not None else ""
    command = f"for f in *.run; do  res=`trec_eval -m P.20 {l_arg} {qrels.resolve().as_posix()} $f`; echo \"$f $res\"; done"
    print(f'Running trec_eval command:\n{command}\nin directory: {run_dir}')

    # Run the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=run_dir.resolve().as_posix())

    if result.stderr:
        print(f"Received command errors: "+result.stderr.strip())
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
            print(f"{method}\t{score}")
            return (method, score)
        else:
            raise RuntimeError(f"Can't parse trec_eval output. Offending line: \"{line}\".\nFull command output:\n{command_output}")

    return dict([parse_line(line.strip()) 
                    for line in command_output.split("\n") 
                    if len(line.strip())>0
                ])

def trec_eval_leaderboard(run_dir:Path, qrels:Path, min_level:Optional[int])-> Dict[str,float]:
    '''Designed to interoperate with `leaderboard_rank_correlation` '''
    output=run_trec_eval(run_dir=run_dir, qrels=qrels, min_level=min_level)
    # output=mimic_trec_eval()
    return parse_trec_eval(output)