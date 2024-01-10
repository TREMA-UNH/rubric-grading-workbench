

import csv
from typing import Set, List, Tuple, Dict, Optional, Any
from pathlib import Path
from collections import defaultdict
import print_correlation_table

from question_types import *
from parse_qrels_runs_with_text import QueryWithFullParagraphList, parseQueryWithFullParagraphs, GradeFilter
from parse_qrels_runs_with_text import *
from exam_cover_metric import *
from exam_cover_metric import compute_exam_cover_scores
import exam_to_qrels
import exam_leaderboard_correlation
import exam_judgment_correlation
from exam_judgment_correlation import ConfusionStats

# for correlation table formatting
def fmt_judgments(js:Set[int])->str:
    return '+'.join([str(j) for j in js])

def fmt_labels(ls:Set[int])->str:
    return '+'.join([str(l) for l in ls]) 


def label_judgments_correlation_table(table_printer:print_correlation_table.TablePrinter
                                      , query_paragraphs: List[QueryWithFullParagraphList], grade_filter:GradeFilter
                                      , predicted_label_list:List[Set[int]], judgment_list:List[Set[int]]
                                      , label_to_judgment_kappa:Dict[str,str]
                                      , judgment_title:Optional[str], label_title:Optional[str]
                                      , min_answers:int=1
                                      ):

    
    # Table Data dictionaries
    counts:Dict[str,Dict[str,int]] = defaultdict(dict) # counts[label][judgment]
    kappas:Dict[str,float] = defaultdict(None)
    
    judgments_header:List[str] = [fmt_judgments(judgment) for judgment in judgment_list]
    label_header:List[str] = [fmt_labels(label) for label in predicted_label_list]

    for label in predicted_label_list:
        for judgment in judgment_list:
            print(f"\n predicted_judgment {label} /  exact_judgment {judgment}")

            corrAll, corrPerQuery = exam_judgment_correlation.confusion_predicted_judgments_correlation(query_paragraphs, grade_filter=grade_filter, judgments=judgment, prediction=label, min_answers=min_answers)
            print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')
            counts[fmt_labels(label)][fmt_judgments(judgment)]=corrAll.predictedRelevant
            if label_to_judgment_kappa[fmt_labels(label)] == fmt_judgments(judgment):
                kappas[fmt_labels(label)] = corrAll.cohen_kappa()

    table_printer.add_table(counts=counts, kappa=kappas
                                        , judgments_header=judgments_header, label_header=label_header
                                        , judgment_title=judgment_title, label_title=label_title
                                        , label_to_judgment_kappa=label_to_judgment_kappa)
        


def run(exam_input_file:Path, qrel_out_file:Path, grade_filter:GradeFilter):
    exam_to_qrels.exam_to_qrels_files(exam_input_file=exam_input_file, qrel_out_file=qrel_out_file, grade_filter=grade_filter)

    # if needed to add more data to the `QueryWithFullParagraphList` objects, write with this function.
    # exam_judgment_correlation.dumpQueryWithFullParagraphList()

    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)
    corrAll:ConfusionStats
    corrPerQuery:Dict[str, ConfusionStats]

    print("ignoring rating levels")

    for min_answers in [1,2,5]:
        for min_judgment_level in [1,2,3]:
            print(f"\n min_judgment {min_judgment_level} / min_answers {min_answers}")

            corrAll, corrPerQuery = exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, min_judgment_level=min_judgment_level, min_answers=min_answers)
            print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')

    # self_rated_correlation_min(grade_filter, query_paragraphs, write_stats=False)
    # self_rated_correlation_exact(grade_filter, query_paragraphs, write_stats=False)



    print("\n\n binary correlation")

    table_printer = print_correlation_table.TablePrinter()
    table_printer.add_section("correlation tables")
        
    for labels in [{0},{1,2,3,4,5}]:
        for judgments in [{0},{1,2,3}]:
            print(f"\n predicted_judgment {labels} /  exact_judgment {judgments}")

            corrAll, corrPerQuery = exam_judgment_correlation.confusion_predicted_judgments_correlation(query_paragraphs, grade_filter=grade_filter, judgments=judgments, prediction=labels, min_answers=1)
            print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')

    def correlation_analysis(min_answers:int):

        table_printer.add_section(f"Min Answers= {min_answers}")
        
        def detailedCorrelation():
            print("\n\n detailed correlation")
        
            predicted_label_list = [{5}, {4}, {3},{2},{1},{0}]
            judgment_list = [{3},{2},{1},{0}]
            
            label_to_judgment_kappa:Dict[str, str]
            label_to_judgment_kappa = { fmt_labels(j):fmt_judgments(j)  for j in judgment_list }
            label_to_judgment_kappa[fmt_labels({5})]=fmt_judgments({2})
            label_to_judgment_kappa[fmt_labels({4})]=fmt_judgments({2})
            label_to_judgment_kappa[fmt_labels({3})]=fmt_judgments({1})
            label_to_judgment_kappa[fmt_labels({2})]=fmt_judgments({1})


            label_judgments_correlation_table(table_printer=table_printer
                                            , query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                            , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                            , label_to_judgment_kappa=label_to_judgment_kappa
                                            ,  judgment_title="Judgments",   label_title="GRADED", min_answers=min_answers)
            table_printer.add_new_paragraph()
        detailedCorrelation()

    
        def mergedCorrelation():
            print("\n\n detailed correlation")
        
            predicted_label_list = [{5,4}, {3,2,1},{0}]
            judgment_list = [{3,2},{1},{0}]
            
            label_to_judgment_kappa:Dict[str, str]
            label_to_judgment_kappa = { fmt_labels(l): fmt_judgments(j) for l,j in zip( predicted_label_list, judgment_list)}


            label_judgments_correlation_table(table_printer=table_printer
                                            , query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                            , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                            , label_to_judgment_kappa=label_to_judgment_kappa
                                            ,  judgment_title="Judgments",   label_title="MERGE", min_answers=min_answers)
            table_printer.add_new_paragraph()
        mergedCorrelation()

        def binaryCorrelation():
            print("\n\n binary correlation")
        
            predicted_label_list = [{3,4,5,1,2},{0}]
            judgment_list = [{3,2,1},{0}]

            label_to_judgment_kappa:Dict[str, str]
            label_to_judgment_kappa = { fmt_labels(l): fmt_judgments(j) for l,j in zip( predicted_label_list, judgment_list)}

            label_judgments_correlation_table(table_printer=table_printer
                                            ,query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                            , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                            , label_to_judgment_kappa=label_to_judgment_kappa
                                            ,  judgment_title="Judgments",   label_title="LENIENT", min_answers=min_answers)
            
            table_printer.add_new_paragraph()
        binaryCorrelation()



        def binaryLenientCorrelation():
            print("\n\n binary correlation")
        
            predicted_label_list = [{3,4,5},{1,2,0}]
            judgment_list = [{3,2,1},{0}]
            
            label_to_judgment_kappa:Dict[str, str]
            label_to_judgment_kappa = { fmt_labels(l): fmt_judgments(j) for l,j in zip( predicted_label_list, judgment_list)}

            label_judgments_correlation_table(table_printer=table_printer
                                            ,query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                            , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                            , label_to_judgment_kappa=label_to_judgment_kappa
                                            ,  judgment_title="Judgments",   label_title="STRICT", min_answers=min_answers)
            
            table_printer.add_new_paragraph()
        binaryLenientCorrelation()

    correlation_analysis(min_answers=1)
    correlation_analysis(min_answers=2)
    correlation_analysis(min_answers=5)

    table_printer.export(Path("./corelation_tables.tex"))

    print("\n\n exam_vs_judged")

    corrAll, corrPerQuery = exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, min_judgment_level=1, min_answers=1)
    for query_id, corr in corrPerQuery.items():
        print(f'{query_id}: examVsJudged {corr.printMeasures()}')# ; manualRankMetric {manualRankMetric.printMeasures()}  ; examRankMetric {examRankMetric.printMeasures()}')
    print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')


    resultsPerMethod = compute_exam_cover_scores(query_paragraphs, grade_filter=grade_filter)
    # resultsPerMethod__ = [val for key, val in resultsPerMethod.items() if key != exam_cover_metric.OVERALL_ENTRY]
    # exam_leaderboard_correlation.print_leaderboard_eval(resultsPerMethod.values())
    exam_leaderboard_correlation.leaderboard_table(resultsPerMethod.values(), grade_filter=grade_filter)

    nExamCorrelation,examCorrelation=exam_leaderboard_correlation.leaderboard_correlation(resultsPerMethod.values())
    print(f' nExam:{nExamCorrelation}')
    print(f' exam:{examCorrelation}')




def self_rated_correlation_exact(grade_filter, query_paragraphs, write_stats:bool=False):
    print("\n")
    print("\n")
    print("\n")

    data = list()
    print("trying different self_rating levels  (exact)")
    for min_answers in [1,2,5]:
        for exact_rating in [0,1,2,3,4,5]:
            for exact_judgment_level in [0,1,2,3]:
                print(f"\n exact_rating {exact_rating} /  exact_judgment {exact_judgment_level} / min_answers {min_answers}")

                corrAll, corrPerQuery = exam_judgment_correlation.confusion_exact_rating_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, exact_judgment_level=exact_judgment_level, min_answers=min_answers, exact_rating=exact_rating)
                print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')
                data.append( {"min_answer": min_answers
                              , "exact_rating": exact_rating
                              , "exact_judgment_level": exact_judgment_level
                              , "kappa": f'{corrAll.cohen_kappa():.2f}'
                              , "tp": corrAll.predictedRelevant
                              , "acc": f'{corrAll.accuracy_measure():.2f}'
                              , "prec": f'{corrAll.prec_measure():.2f}'
                              , "rec": f'{corrAll.rec_measure():.2f}'
                            })

    if write_stats:
        headers = ["min_answer", "exact_rating", "exact_judgment_level", "tp", "kappa", "acc", "prec", "rec"]

        file_path = "exact_rating_correlation.tsv"

        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter='\t')
            writer.writeheader()  # Write the header automatically
            writer.writerows(data)

def self_rated_correlation_min(grade_filter, query_paragraphs, write_stats=False):
    print("\n")
    print("\n")
    print("\n")

    data = list()
    print("trying different self_rating levels  (>=)")
    for min_answers in [1,2,5]:
        for min_rating in [1,2,3,4,5]:
            for min_judgment_level in [1,2,3]:
                print(f"\n min_rating {min_rating} /  min_judgment {min_judgment_level} / min_answers {min_answers}")

                corrAll, corrPerQuery = exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, min_judgment_level=min_judgment_level, min_answers=min_answers, min_rating=min_rating)
                print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')
                data.append( {"min_answer": min_answers
                              , "min_rating": min_rating
                              , "min_judgment_level": min_judgment_level
                              , "kappa": f'{corrAll.cohen_kappa():.2f}'
                              , "tp": corrAll.predictedRelevant
                              , "acc": f'{corrAll.accuracy_measure():.2f}'
                              , "prec": f'{corrAll.prec_measure():.2f}'
                              , "rec": f'{corrAll.rec_measure():.2f}'
                            })

            for exact_judgment_level in [0]:
                print(f"\n exact_rating {min_rating} /  exact_judgment {min_judgment_level} / min_answers {min_answers}")

                corrAll, corrPerQuery = exam_judgment_correlation.confusion_exact_rating_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, exact_judgment_level=exact_judgment_level, min_answers=min_answers, min_rating=min_rating)               
                print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')
                data.append( {"min_answer": min_answers
                              , "min_rating": min_rating
                              , "min_judgment_level": exact_judgment_level
                              , "kappa": f'{corrAll.cohen_kappa():.2f}'
                              , "tp": corrAll.predictedRelevant
                              , "acc": f'{corrAll.accuracy_measure():.2f}'
                              , "prec": f'{corrAll.prec_measure():.2f}'
                              , "rec": f'{corrAll.rec_measure():.2f}'
                            })

    if write_stats:
        headers = ["min_answer", "min_rating", "min_judgment_level", "tp", "kappa", "acc", "prec", "rec"]

        file_path = "min_rating_correlation.tsv"
        
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter='\t')
            writer.writeheader()  # Write the header automatically
            writer.writerows(data)
            print("\n")


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
    parser.add_argument('--prompt-class', type=str, choices=get_prompt_classes(), required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))

    # Parse the arguments
    args = parser.parse_args()    
    grade_filter = GradeFilter(model_name=args.model, prompt_class = args.prompt_class, is_self_rated=True, min_self_rating=None)
    run(exam_input_file=args.exam_annotated_file, qrel_out_file=args.qrel_out, grade_filter=grade_filter)




if __name__ == "__main__":
    main()
