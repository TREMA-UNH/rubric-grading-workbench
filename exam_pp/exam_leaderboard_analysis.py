

import csv
import sys
from typing import Set, List, Tuple, Dict, Optional, Any
from pathlib import Path
from collections import defaultdict


from .test_bank_prompts import get_prompt_classes, get_prompt_type_from_prompt_class
from .data_model import FullParagraphData, QueryWithFullParagraphList, parseQueryWithFullParagraphs, GradeFilter
from .exam_cover_metric import ExamCoverEvals, ExamCoverScorerFactory, compute_exam_cover_scores
from .exam_judgment_correlation import ConfusionStats
from .exam_run_trec_eval import trec_eval_leaderboard
from . exam_post_pipeline import export_qrels, qrel_leaderboard_analysis, print_correlation_table, exam_judgment_correlation, exam_leaderboard_correlation, exam_to_qrels


def main(cmdargs=None):
    import argparse

    sys.stdout.reconfigure(line_buffering=True)


    print("EXAM Leaderboard Analysis Pipeline")
    desc = f'''EXAM Leaderboard Analysis Pipeline \n
             '''
    

    parser = argparse.ArgumentParser(description="EXAM leaderboard analysis"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exam_annotated_file', type=str, metavar='exam-xxx.jsonl.gz'
                        , help='json file that annotates each paragraph with a number of answerable questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )
    # parser.add_argument('-q', '--qrel-out', type=str, metavar="FILE", help='Export Qrels to this file', default=None)
    parser.add_argument('--qrel-dir', type=str, metavar="DIR", help='Directory to write qrel files to. Default "./" ', default=".")
    parser.add_argument('--qrel-analysis-out', type=str, metavar="FILE", help='Export Leaderboard analysis to this file. (applies only to -q)',  required=True)
    parser.add_argument('--qrel-query-facets', action='store_true', help='If set, will use query facets for qrels (prefix of question_ids). (applies only to -q)', default=None)
    parser.add_argument('--run-dir', type=str, metavar="DIR", help='Directory of trec_eval run-files. These must be uncompressed, the filename must match the pattern "${methodname}.run" where methodname refers to the method name in the official leaderboard. If set, will use the exported qrel file to determine correlation with the official leaderboard. (applies only to -q)', required=True)
    parser.add_argument('--trec-eval-qrel-correlation',  type=str, metavar="IN-FILE", help='Will use this qrel file to measure leaderboard correlation with trec_eval (applies only to -q)', default=None)
    # parser.add_argument('--min-trec-eval-level',  type=int, metavar="LEVEL", help='Relevance cutoff level for trec_eval. If not set, multiple levels will be tried (applies only to -q)', default=1)
    parser.add_argument('--trec-eval-metric', type=str, metavar="str", help='Which evaluation metric to use in trec_eval. (applies only to -q)', nargs='+')



    parser.add_argument('-m', '--model', type=str, metavar="HF_MODEL_NAME", help='the hugging face model name used by the Q/A module.')
    parser.add_argument('--prompt-class', type=str, choices=get_prompt_classes(), required=True, nargs='+', metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))
    parser.add_argument('-r', '--use-ratings', action='store_true', help='If set, correlation analysis will use graded self-ratings. Default is to use the number of correct answers.')
    # parser.add_argument('--use-relevance-prompt', action='store_true', help='If set, use relevance prompt instead of exam grades. (Inter-annotator only)')
    parser.add_argument('--min-self-rating', type=int, metavar="RATING", help='If set, will only count ratings >= RATING as relevant for leaderboards. (Only applies to when -r is used.)')
    
    parser.add_argument('--question-set', type=str, choices=["tqa","genq","question-bank"], metavar="SET ", help='Which question set to use. Options: tqa, genq,  or question-bank ')
    parser.add_argument('--official-leaderboard', type=str, metavar="JSON-FILE", help='Use leaderboard JSON file instead (format {"methodName":rank})', default=None)
    parser.add_argument('--min-relevant-judgment', type=int, default=1, metavar="LEVEL", help='Minimum judgment levelfor relevant passages. (Set to 2 for TREC DL)')
    parser.add_argument('--testset', type=str, choices=["cary3","dl19","dl20"], metavar="SET", help='Offers hard-coded defaults for --official-leaderboard and --min-relevant-judgment for some test sets. Options: cary3, dl19, or dl20 ')
    

    # Parse the arguments
    if cmdargs is not None:
        args = parser.parse_args(args=cmdargs)    
    else:
        args = parser.parse_args()
        
    # grade_filter = GradeFilter(model_name=args.model, prompt_class = args.prompt_class, is_self_rated=None, min_self_rating=None, question_set=args.question_set, prompt_type=get_prompt_type_from_prompt_class(args.prompt_class))


    print(f"args: {args.prompt_class}  {args.trec_eval_metric}")

    exam_input_file=args.exam_annotated_file
    use_ratings=args.use_ratings

    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)



    official_leaderboard:Dict[str,float]
    non_relevant_grades = None
    relevant_grades = None
    if args.official_leaderboard is not None:
        official_leaderboard = exam_leaderboard_correlation.load_leaderboard(args.official_leaderboard)
    elif args.testset == "cary3":
        official_leaderboard = exam_leaderboard_correlation.official_CarY3_leaderboard 
    elif args.testset == "dl19":
        official_leaderboard = exam_leaderboard_correlation.official_DL19_Leaderboard
        non_relevant_grades = {0,1}
        relevant_grades = {2,3}
    elif args.testset == "dl20":
        official_leaderboard = exam_leaderboard_correlation.official_DL20_Leaderboard
        non_relevant_grades = {0,1}
        relevant_grades = {2,3}

    relevant_grades = set(range(args.min_relevant_judgment, 4))
    non_relevant_grades = set(range(-2, args.min_relevant_judgment))

    # if args.qrel_analysis_out is not None and args.run_dir is not None:

    # for each method, produce qrels
    qrel_files:List[Path] = list()

    for prompt_class in args.prompt_class:
        grade_filter = GradeFilter(model_name=args.model, prompt_class = prompt_class, is_self_rated=None, min_self_rating=None, question_set=args.question_set, prompt_type=get_prompt_type_from_prompt_class(prompt_class))
        qrel_file_name = f"{args.qrel_dir}/{prompt_class}.qrel"

        export_qrels(query_paragraphs=query_paragraphs
                    , qrel_out_file=qrel_file_name
                    , grade_filter=grade_filter
                    , use_query_facets=args.qrel_query_facets
                    , use_ratings=args.use_ratings
                    )
        qrel_files.append(Path(qrel_file_name))


    qrel_leaderboard_analysis(qrels_files=qrel_files
                            , run_dir=Path(args.run_dir)
                            , min_levels=[1,3,4,5]
                            , official_leaderboard=official_leaderboard
                            , analysis_out=args.qrel_analysis_out
                            , trec_eval_metrics=args.trec_eval_metric # ["ndcg_cut.10", "map", "recip_rank"]
                            )
        

if __name__ == "__main__":
    main()
