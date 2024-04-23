

import csv
import sys
from typing import Set, List, Tuple, Dict, Optional, Any
from pathlib import Path
from collections import defaultdict

import exam_grading
import question_bank_loader
import tqa_loader

from .exam_post_pipeline import run_minimal_qrel_leaderboard, run_qrel_leaderboard

from .exam_to_qrels import exam_to_qrels_files
from . import exam_to_qrels


from .test_bank_prompts import DirectGradingPrompt, get_prompt_classes, get_prompt_type_from_prompt_class
from .data_model import FullParagraphData, QueryWithFullParagraphList, parseQueryWithFullParagraphs, GradeFilter
from .exam_cover_metric import ExamCoverEvals, ExamCoverScorerFactory, compute_exam_cover_scores
# from . import exam_to_qrels
# from . import exam_leaderboard_correlation
# from . import exam_judgment_correlation
# from .exam_judgment_correlation import ConfusionStats
# from .exam_run_trec_eval import trec_eval_leaderboard
# from . import print_correlation_table




def export_leaderboard_table(leaderboard_out:Path, evals:List[ExamCoverEvals],sortBy:Optional[str]=None):
    '''Latest version'''
    evals_ = sorted (evals, key= lambda eval: eval.method)
    if sortBy == "exam":
        evals_ = sorted(evals, key= lambda eval: eval.examScore, reverse=True)

    def f2s(x:Optional[float])->str:
        if x is None:
            return ' '
        else:
            return f'{x:.3f}'
    header = '\t'.join(['method'
                        ,'ExamCover','+/-','std-err'
                        ])

    lines = [ '\t'.join([e.method
                        ,f2s(e.examScore), '+/-', f2s(e.examScoreStd)
                        ])
                    for e in evals_]
    
    with open(leaderboard_out, 'wt') as file:
        file.write('\n'.join([header]+lines))
        file.close()
    print(f"ExamCover Leaderboard written to {leaderboard_out}")
    



def run_leaderboard(leaderboard_file:Path, grade_filter:GradeFilter, query_paragraphs, min_self_rating: Optional[int]=1):
    exam_factory = ExamCoverScorerFactory(grade_filter=grade_filter, min_self_rating=min_self_rating)
    resultsPerMethod:Dict[str, ExamCoverEvals] = compute_exam_cover_scores(query_paragraphs, exam_factory=exam_factory)
    export_leaderboard_table(leaderboard_out=leaderboard_file, evals= list(resultsPerMethod.values()), sortBy="exam")


# -----------------------
#  Qrel
    


# Todo Sync with exam_post_pipeline.py #export_qrels
def export_qrels(query_paragraphs,  qrel_out_file:Path, grade_filter:GradeFilter, use_query_facets:bool = False, direct_grading:bool = False, use_ratings:bool = False, query_facets: Dict[str,Set[str]]=dict()):
    if use_query_facets:
        if direct_grading:
            if use_ratings:
                qrel_entries = exam_to_qrels.convert_direct_to_rated_facet_qrels(query_paragraphs,grade_filter=grade_filter, query_facets=query_facets)
            else:
                qrel_entries = exam_to_qrels.convert_direct_to_facet_qrels(query_paragraphs,grade_filter=grade_filter, query_facets=query_facets)

        else:
            if use_ratings:
                qrel_entries = exam_to_qrels.convert_exam_to_rated_facet_qrels(query_paragraphs,grade_filter=grade_filter)
            else:
                qrel_entries = exam_to_qrels.convert_exam_to_facet_qrels(query_paragraphs,grade_filter=grade_filter)
    else:
        if use_ratings:
            qrel_entries = exam_to_qrels.convert_exam_to_rated_qrels(query_paragraphs,grade_filter=grade_filter)
        else:
            qrel_entries = exam_to_qrels.conver_exam_to_qrels(query_paragraphs,grade_filter=grade_filter)

    exam_to_qrels.write_qrel_file(qrel_out_file, qrel_entries)



# --------------------





def main(cmdargs=None):
    import argparse

    sys.stdout.reconfigure(line_buffering=True)


    print("EXAM Post Pipeline")
    desc = f'''EXAM Post Pipeline \n
              The input file (i.e, exam_annotated_file) has to be a *JSONL.GZ file that follows this structure: \n
              \n  
                  [query_id, [FullParagraphData]] \n
              \n
               where `FullParagraphData` meets the following structure \n
             {FullParagraphData.schema_json(indent=2)}
             '''
    

    parser = argparse.ArgumentParser(description="EXAM pipeline"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exam_graded_file', type=str, metavar='exam-xxx.jsonl.gz'
                        , help='json file that annotates each paragraph with a number of answerable questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )
    parser.add_argument('-q', '--qrel-out', type=str, metavar="FILE", help='Export Qrels to this file', default=None)
    parser.add_argument('--qrel-leaderboard-out', type=str, metavar="FILE", help='Export Exam-Qrels leaderboard to this file', default=None)
    parser.add_argument('--trec-eval-metric', type=str, metavar="str", help='Which evaluation metric to use in trec_eval. Default: P.20. (applies only to -q)', default="P.20")
    parser.add_argument('--qrel-query-facets', action='store_true', help='If set, will use query facets for qrels (prefix of question_ids)', default=None)
    parser.add_argument('--run-dir', type=str, metavar="DIR", help='Directory of trec_eval run-files. These must be uncompressed, the filename must match the pattern "${methodname}.run" where methodname refers to the method name in the official leaderboard. If set, will use the exported qrel file to determine correlation with the official leaderboard', default=None)
    # parser.add_argument('--trec-eval-qrel-correlation',  type=str, metavar="IN-FILE", help='Will use this qrel file to measure leaderboard correlation with trec_eval', default=None)
    # parser.add_argument('--min-trec-eval-level',  type=int, metavar="LEVEL", help='Relevance cutoff level for trec_eval. If not set, multiple levels will be tried', default=None)

    # parser.add_argument('--correlation-out', type=str, metavar="FILE", help='Export Inter-annotator Agreement Correlation to this file ', default=None)

    parser.add_argument('--leaderboard-out', type=str, metavar="FILE", help='Export Leaderboard to this file ', default=None)

    parser.add_argument('-m', '--model', type=str, metavar="HF_MODEL_NAME", help='the hugging face model name used by the Q/A module.')
    parser.add_argument('--prompt-class', type=str, choices=get_prompt_classes(), required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))
    parser.add_argument('-r', '--use-ratings', action='store_true', help='If set, will use  graded self-ratings. Default is to use the number of correct answers.')
    # parser.add_argument('--use-relevance-prompt', action='store_true', help='If set, use relevance prompt instead of exam grades. (Inter-annotator only)')
    parser.add_argument('--min-self-rating', type=int, metavar="RATING", help='If set, will use self-ratings  >= RATING as relevant. (only applies to exam grades with self-ratings)')
    parser.add_argument('--question-set', type=str, choices=["tqa","genq","question-bank"], metavar="SET ", help='Which question set to use. Options: tqa or naghmeh ')
    parser.add_argument('--question-path', type=str, metavar='PATH', help='Path to read exam questions from (can be tqa directory or question-bank file) -- only needed for direct grading with facets')
    # parser.add_argument('--testset', type=str, choices=["cary3","dl19"], required=True, metavar="SET ", help='Which question set to use. Options: tqa or naghmeh ')
    # parser.add_argument('--official-leaderboard', type=str, metavar="JSON-FILE", help='Use leaderboard JSON file instead (format {"methodName":rank})', default=None)
    

    # Parse the arguments
    args = parser.parse_args(args=cmdargs)
    grade_filter = GradeFilter(model_name=args.model, prompt_class = args.prompt_class, is_self_rated=None, min_self_rating=None, question_set=args.question_set, prompt_type=get_prompt_type_from_prompt_class(args.prompt_class))


    exam_input_file=args.exam_graded_file


    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)



    query_facets:Dict[str,Set[str]] = {}
    if args.qrel_query_facets and get_prompt_type_from_prompt_class(args.prompt_class)==DirectGradingPrompt.my_prompt_type:
        # we have to load the questions and get facets for each query
        # so we can emit facet-based query information with the qrel file

        print("Loading query facets for direct grading qrels")

        if args.question_set == "tqa":
            tqabank = tqa_loader.load_TQA_questions(tqa_file=args.question_path)
            for query_id, tqa_questions in tqabank:
                if not query_id in query_facets:
                    query_facets[query_id]=set()
                for tqa_question in tqa_questions:
                    query_facets[query_id].add(tqa_question.facet_id)

        elif args.question_set == 'question-bank':
            testbank = question_bank_loader.parseTestBank(file_path=args.question_path, use_nuggets=False)
            for bank in testbank:
                if not bank.query_id in query_facets:
                    query_facets[bank.query_id]=set()
                query_facets[bank.query_id].add(bank.facet_id)

        else:
            raise f"loading of facets for question set {args.question_path} is not implemented"




    if args.qrel_out is not None:
        export_qrels(query_paragraphs=query_paragraphs
                     , qrel_out_file=args.qrel_out
                     , grade_filter=grade_filter
                     , use_query_facets=args.qrel_query_facets
                     , use_ratings=args.use_ratings
                     , query_facets=query_facets
                     , direct_grading= args.qrel_query_facets and get_prompt_type_from_prompt_class(args.prompt_class)==DirectGradingPrompt.my_prompt_type
                     )
        print("qrel leaderboard")

        if args.run_dir is not None:
            run_minimal_qrel_leaderboard(qrels_file=Path(args.qrel_out)
                                 ,run_dir=Path(args.run_dir)
                                 , min_level=args.min_self_rating
                                 , leaderboard_out=args.qrel_leaderboard_out
                                 , trec_eval_metric=args.trec_eval_metric
                                 )


    if args.leaderboard_out is not None:
        run_leaderboard(leaderboard_file=args.leaderboard_out
                        , grade_filter=grade_filter
                        , query_paragraphs=query_paragraphs
                        , min_self_rating=args.min_self_rating
                        )



if __name__ == "__main__":
    main()
