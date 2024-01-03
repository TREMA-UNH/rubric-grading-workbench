

from question_types import *
from parse_qrels_runs_with_text import *
from typing import Set, List, Tuple

from exam_cover_metric import *
import exam_to_qrels
import exam_leaderboard_correlation
import exam_judgment_correlation

def run(exam_input_file:Path, qrel_out_file:Path):
    exam_to_qrels.exam_to_qrels_files(exam_input_file=exam_input_file, qrel_out_file=qrel_out_file)

    # if needed to add more data to the `QueryWithFullParagraphList` objects, write with this function.
    # exam_judgment_correlation.dumpQueryWithFullParagraphList()

    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)
    exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, min_judgment_level=1, min_answers=1)


    resultsPerMethod = compute_exam_cover_scores(query_paragraphs)
    # resultsPerMethod__ = [val for key, val in resultsPerMethod.items() if key != exam_cover_metric.OVERALL_ENTRY]
    exam_leaderboard_correlation.print_leaderboard_eval(resultsPerMethod.values())

    nExamCorrelation,examCorrelation=exam_leaderboard_correlation.leaderboard_correlation(resultsPerMethod.values())
    print(f' nExam:{nExamCorrelation}')
    print(f' exam:{examCorrelation}')

def main():
    import argparse

    desc = f'''EXAM Pipeline \n
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

    # Parse the arguments
    args = parser.parse_args()    
    run(exam_input_file=args.exam_annotated_file, qrel_out_file=args.qrel_out)




if __name__ == "__main__":
    main()
