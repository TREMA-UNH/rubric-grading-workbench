

from question_types import *
from parse_qrels_runs_with_text import QueryWithFullParagraphList, parseQueryWithFullParagraphs
from parse_qrels_runs_with_text import *
from typing import Set, List, Tuple, Dict, Optional, Any
from pathlib import Path


from exam_cover_metric import *
from exam_cover_metric import compute_exam_cover_scores
import exam_to_qrels
import exam_leaderboard_correlation
import exam_judgment_correlation
from exam_judgment_correlation import ConfusionStats


def run(exam_input_file:Path, qrel_out_file:Path, model_name:str):
    exam_to_qrels.exam_to_qrels_files(exam_input_file=exam_input_file, qrel_out_file=qrel_out_file, model_name=model_name)

    # if needed to add more data to the `QueryWithFullParagraphList` objects, write with this function.
    # exam_judgment_correlation.dumpQueryWithFullParagraphList()

    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)
    corrAll:ConfusionStats
    corrPerQuery:Dict[str, ConfusionStats]

    # for min_answers in [1,2,5]:
    #     for min_judgment_level in [1,2,3]:
    #         print(f"\n min_judgment {min_judgment_level} / min_answers {min_answers}")

    #         corrAll, corrPerQuery = exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, model_name=model_name, min_judgment_level=min_judgment_level, min_answers=min_answers)
    #         print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')

    # print("\n")

    corrAll, corrPerQuery = exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, model_name=model_name, min_judgment_level=1, min_answers=1)
    for query_id, corr in corrPerQuery.items():
        print(f'{query_id}: examVsJudged {corr.printMeasures()}')# ; manualRankMetric {manualRankMetric.printMeasures()}  ; examRankMetric {examRankMetric.printMeasures()}')
    print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')


    resultsPerMethod = compute_exam_cover_scores(query_paragraphs, model_name=model_name)
    # resultsPerMethod__ = [val for key, val in resultsPerMethod.items() if key != exam_cover_metric.OVERALL_ENTRY]
    # exam_leaderboard_correlation.print_leaderboard_eval(resultsPerMethod.values())
    exam_leaderboard_correlation.leaderboard_table(resultsPerMethod.values(), model_name = model_name)

    nExamCorrelation,examCorrelation=exam_leaderboard_correlation.leaderboard_correlation(resultsPerMethod.values())
    print(f' nExam:{nExamCorrelation}')
    print(f' exam:{examCorrelation}')

def main():
    import argparse

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
    parser.add_argument('exam_annotated_file', type=str, metavar='exam-xxx.jsonl.gz'
                        , help='json file that annotates each paragraph with a number of anserable questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )

    parser.add_argument('-q', '--qrel-out', type=str, metavar="FILE", help='Output QREL file name', default='output.qrels')
    parser.add_argument('-m', '--model', type=str, metavar="HF_MODEL_NAME", help='the hugging face model name used by the Q/A module.')

    # Parse the arguments
    args = parser.parse_args()    
    run(exam_input_file=args.exam_annotated_file, qrel_out_file=args.qrel_out, model_name=args.model)




if __name__ == "__main__":
    main()
