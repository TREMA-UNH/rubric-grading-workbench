    # Compute Representative queries for trec_eval leaderboard
    
    # Runs trec_eval, then identifies a small set of queries that by
    # themselves most closely obtain a leaderboard (system ranking) that is
    # most closely to the leaderboard obtained when running on all queries


from pathlib import Path
from typing import Dict,Tuple, List
import sys 

import scipy
from scipy.stats import spearmanr, kendalltau, rankdata
import scipy.stats
from . import exam_run_trec_eval



def compatible_kendalltau(ranks1, ranks2)->float:
    from packaging import version

    if version.parse(scipy.__version__) >= version.parse('1.7.0'):    
    # if scipy.__version__ >= '1.7.0':
        # For scipy 1.7.0 and later
        tau, p_value = kendalltau(ranks1, ranks2)
        return tau
    else:
        # For older versions
        from scipy.stats import SignificanceResult
        result = kendalltau(ranks1, ranks2)
        return result.correlation


def representative_queries(method_per_query_results:Dict[str,List[Tuple[str,float]]], num_rep:int)->List[str]:
    methods = list(method_per_query_results.keys())
    query_set = list({query for method, data in method_per_query_results.items() for query,score in data})
    
    method_per_query_dict = {method: {query: score for query,score in data} for method, data in method_per_query_results.items()}


    def system_ranking(query_id:str):
        return rankdata([-method_per_query_dict[method][query_id] for method in methods]) # negative scores for descending order

    query_representation:List[Tuple[str,float]] = list()
    overall_rankdata = system_ranking("all")
    for query_id in query_set:
        if query_id != "all":
            query_rankdata = system_ranking(query_id)
            representation = compatible_kendalltau(overall_rankdata, query_rankdata)
            # print(f"{query_id}: {representation}")
            query_representation.append((query_id, representation))

    query_representation = sorted(query_representation, key=lambda x:x[1], reverse=True)
    # print(query_representation[:num_rep])
    print("Queries ordered by representativeness")
    for q,r in query_representation:
        print(f"{q}: {r}")
    return [q for q, r in query_representation[0:num_rep]]



def trec_eval_representative_queries(run_dir:Path, qrels_file:Path, min_level:int, trec_eval_metric:str, num_rep:int)->List[str]:
    trec_eval_out =exam_run_trec_eval.run_trec_eval_variance(run_dir=run_dir, qrels=qrels_file, min_level=min_level, trec_eval_metric=trec_eval_metric, trec_eval_args="-c")
    
    # print("----\n",trec_eval_out,"\n----")
    trec_eval_parsed=exam_run_trec_eval.parse_trec_eval_per_query(trec_eval_out)
    rep_queries = representative_queries(method_per_query_results=trec_eval_parsed, num_rep=num_rep)
    print(f"Best Queries: {rep_queries}")
    return rep_queries


def main(cmdargs=None):
    import argparse

    sys.stdout.reconfigure(line_buffering=True)


    print("Most representative queries")
    desc = f'''Utility to identify most reprentative queries for a system evaluation. \n

              Runs trec_eval, then identifies a small set of queries that by
              themselves most closely obtain a leaderboard (system ranking) that is
              most closely to the leaderboard obtained when running on all queries

              The resemblance is measured with Kendall's Tau.
             '''
    

    parser = argparse.ArgumentParser(description="EXAM pipeline"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-q', '--qrel-file', type=str, metavar="FILE", help='Qrels to determine system ranking', required=True)
    parser.add_argument('--trec-eval-metric', type=str, metavar="str", help='Which evaluation metric to use in trec_eval.', required=True)
    parser.add_argument('--run-dir', type=str, metavar="DIR", help='Directory of trec_eval run-files. These must be uncompressed, the filename must match the pattern "${methodname}.run" where methodname refers to the method name in the official leaderboard. If set, will use the exported qrel file to determine correlation with the official leaderboard', required=True)
    parser.add_argument('--min-relevance-level',  type=int, metavar="LEVEL", help='Relevance cutoff level for trec_eval.', default=1)
    parser.add_argument('--num-representative',  type=int, metavar="N", help='Number of most representative queries to be emitted', required=True)



    # Parse the arguments
    args = parser.parse_args(args=cmdargs)

    trec_eval_representative_queries(run_dir=Path(args.run_dir)
                                     , qrels_file= Path(args.qrel_file).absolute()
                                     , min_level= args.min_relevance_level
                                     , trec_eval_metric= args.trec_eval_metric
                                     , num_rep= args.num_representative)

if __name__ == "__main__":
    main()
