

import csv
import sys
from typing import Set, List, Tuple, Dict, Optional, Any
from pathlib import Path
from collections import defaultdict


from .question_types import get_prompt_classes
from .parse_qrels_runs_with_text import FullParagraphData, QueryWithFullParagraphList, parseQueryWithFullParagraphs, GradeFilter
from .exam_cover_metric import ExamCoverEvals, ExamCoverScorerFactory, compute_exam_cover_scores
from . import exam_to_qrels
from . import exam_leaderboard_correlation
from . import exam_judgment_correlation
from .exam_judgment_correlation import ConfusionStats
from .exam_run_trec_eval import trec_eval_leaderboard
from . import print_correlation_table

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
                                      , use_ratings:bool
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

            corrAll, corrPerQuery = exam_judgment_correlation.confusion_predicted_judgments_correlation(query_paragraphs
                                                                                                        , grade_filter=grade_filter
                                                                                                        , judgments=judgment
                                                                                                        , prediction=label
                                                                                                        , min_answers=min_answers
                                                                                                        ,use_ratings=use_ratings)
            print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')
            counts[fmt_labels(label)][fmt_judgments(judgment)]=corrAll.predictedRelevant
            if label_to_judgment_kappa[fmt_labels(label)] == fmt_judgments(judgment):
                kappas[fmt_labels(label)] = corrAll.cohen_kappa()

    table_printer.add_table(counts=counts, kappa=kappas
                                        , judgments_header=judgments_header, label_header=label_header
                                        , judgment_title=judgment_title, label_title=label_title
                                        , label_to_judgment_kappa=label_to_judgment_kappa)
        

def export_qrels(query_paragraphs,  qrel_out_file:Path, grade_filter:GradeFilter, use_query_facets:bool = False, use_ratings:bool = False):
    if use_query_facets:
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



def run_interannotator_agreement(correlation_out_file:Path, grade_filter, use_ratings, query_paragraphs
                                 , relevant_grades:Optional[Set[int]]=None,  non_relevant_grades:Optional[Set[int]]=None):
    corrAll:ConfusionStats
    corrPerQuery:Dict[str, ConfusionStats]

    print("ignoring rating levels")

    non_relevant_grade_set = {0}
    relevant_grade_set = {1,2,3}

    if non_relevant_grades is not None:
        non_relevant_grade_set = non_relevant_grades
    if relevant_grades is not None:
        relevant_grade_set = relevant_grades


    for min_answers in [1,2,5]:
        for min_judgment_level in relevant_grade_set:
            print(f"\n min_judgment {min_judgment_level} / min_answers {min_answers}")

            corrAll, corrPerQuery = exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, min_judgment_level=min_judgment_level, min_answers=min_answers)
            print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')

    # self_rated_correlation_min(grade_filter, query_paragraphs, write_stats=False)
    # self_rated_correlation_exact(grade_filter, query_paragraphs, write_stats=False)


    if use_ratings:
        selfRated_vs_judged_correlation(correlation_out_file, grade_filter, query_paragraphs, relevant_grade_set, non_relevant_grade_set)
    else:
        binary_vs_judged_correlation(correlation_out_file, grade_filter, query_paragraphs, relevant_grade_set, non_relevant_grade_set)

    print("\n\n exam_vs_judged")

    corrAll, corrPerQuery = exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, min_judgment_level=1, min_answers=1)
    for query_id, corr in corrPerQuery.items():
        print(f'{query_id}: examVsJudged {corr.printMeasures()}')# ; manualRankMetric {manualRankMetric.printMeasures()}  ; examRankMetric {examRankMetric.printMeasures()}')
    print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')

def run_leaderboard(leaderboard_file:Path, grade_filter:GradeFilter, query_paragraphs, use_ratings:bool,   official_leaderboard:Dict[str,int], min_self_rating = Optional[int]):
    with open(leaderboard_file, 'wt') as file:
        min_rating:Optional[int]

        for min_rating in ([min_self_rating] if min_self_rating is not None else ([1,2,3,4,5] if use_ratings else [None])):
            exam_factory = ExamCoverScorerFactory(grade_filter=grade_filter, min_self_rating=min_rating)
            resultsPerMethod:Dict[str, ExamCoverEvals] = compute_exam_cover_scores(query_paragraphs, exam_factory=exam_factory)
            # resultsPerMethod__ = [val for key, val in resultsPerMethod.items() if key != exam_cover_metric.OVERALL_ENTRY]
            exam_leaderboard_correlation.print_leaderboard_eval(list(resultsPerMethod.values()), grade_filter=grade_filter)

            nExamCorrelation,examCorrelation=exam_leaderboard_correlation.leaderboard_correlation(resultsPerMethod.values(), official_leaderboard=official_leaderboard)
            print(f'min_rating={str(min_rating)} nExam:{nExamCorrelation}')
            print(f'min_rating={str(min_rating)}  exam:{examCorrelation}')

            table = exam_leaderboard_correlation.leaderboard_table(list(resultsPerMethod.values())
                                                                   , official_leaderboard=official_leaderboard
                                                                   , nExamCorrelation=nExamCorrelation
                                                                   , examCorrelation=examCorrelation) 
            


        
            file.writelines("\n".join(table))
            file.writelines( ["\n"
                            , f' EXAM scores produced with {grade_filter}\n'
                            , f' min_rating\t{str(min_rating)}\n'
                            ,'\n'])

            file.writelines(["\n","\n"])

        file.close()

def run_qrel_leaderboard(qrels_file:Path, run_dir:Path,  official_leaderboard:Dict[str,int], leaderboard_out:Path, min_level = Optional[int]):
    with open(leaderboard_out, 'wt') as file:
        file.write("based on Exam-Qrels\n")

        for min_level_x in ([1,2,3,4,5] if min_level is None else [min_level]):

            print(f'run_dir={run_dir}\n qrels_file={qrels_file}\nmin_level={min_level_x}')
            methodScores = trec_eval_leaderboard(run_dir=run_dir, qrels=qrels_file, min_level=min_level_x)
            correlationStats=exam_leaderboard_correlation.leaderboard_rank_correlation(methodScores, official_leaderboard=official_leaderboard)

            examScores = [
                exam_leaderboard_correlation.ExamCoverEvals(method=method
                                                        , examScore=score
                                                        , nExamScore=0
                                                        , examScoreStd=0
                                                        , nExamScoreStd=0
                                                        , examCoverPerQuery={}
                                                        , nExamCoverPerQuery={} 
                                                        ) 
                        for method, score in methodScores.items()]
            lines = exam_leaderboard_correlation.leaderboard_table(evals =examScores
                                    , official_leaderboard=official_leaderboard 
                                    , nExamCorrelation = None
                                    , examCorrelation=correlationStats
                                    )
            file.write('\n'.join(lines))
            file.write('\n'.join([""
                                  ,f"min_rating\t{min_level_x:.0f}"
                                , f"qrel_file\t{qrels_file}"
                                , "\n"]))
        
            print(f'min_level\t{min_level_x}\tcorrelation\t{correlationStats.pretty_print()}\n')
            # file.writelines("\n".join(table))
            # file.writelines( ["\n"
            #                 # , f' EXAM scores produced with {grade_filter}\n'
            #                 # , f' min_rating\t{str(min_rating)}\n'
            #                 , f' nExam\t{nExamCorrelation.pretty_print()}\n'
            #                 , f' exam\t{examCorrelation.pretty_print()}\n'
            #                 ,'\n'])

        file.writelines(["\n","\n"])
    file.close()
    print(f"exam-qrels leaderboard written to {leaderboard_out}")

def binary_vs_judged_correlation(correlation_out_file:Path, grade_filter:GradeFilter, query_paragraphs
                                 , relevant_grade_set:Set[int],  non_relevant_grade_set:Set[int]):
    print("\n\n binary correlation")
   
    table_printer = print_correlation_table.TablePrinter()
    table_printer.add_section("correlation tables")


    def binaryCorrelation(min_answers:int):
        print("\n\n binary correlation")
    
        predicted_label_list = [{1},{0}]
        judgment_list = [relevant_grade_set, non_relevant_grade_set]

        label_to_judgment_kappa:Dict[str, str]
        label_to_judgment_kappa = { fmt_labels(l): fmt_judgments(j) for l,j in zip( predicted_label_list, judgment_list)}

        label_judgments_correlation_table(table_printer=table_printer
                                        ,query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                        , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                        , label_to_judgment_kappa=label_to_judgment_kappa
                                        ,  judgment_title="Judgments",   label_title="BINARY"
                                        , min_answers=min_answers, use_ratings=False)
        
        table_printer.add_new_paragraph()



    def detailedCorrelation(min_answers:int):
        print("\n\n detailed correlation")
    
        predicted_label_list = [{1},{0}]
        judgment_list = [{3},{2},{1},{0}]
        
        label_to_judgment_kappa:Dict[str, str]={}
        # label_to_judgment_kappa = { fmt_labels(j):fmt_judgments(j)  for j in judgment_list }
        label_to_judgment_kappa[fmt_labels({1})]=fmt_judgments({2})
        label_to_judgment_kappa[fmt_labels({0})]=fmt_judgments({0})


        label_judgments_correlation_table(table_printer=table_printer
                                        , query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                        , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                        , label_to_judgment_kappa=label_to_judgment_kappa
                                        ,  judgment_title="Judgments",   label_title="GRADED"
                                        , min_answers=min_answers, use_ratings=False)
        table_printer.add_new_paragraph()
    
    
    table_printer.add_section(f"min answers 1")
    binaryCorrelation(min_answers=1)
    detailedCorrelation(min_answers=1)

    table_printer.add_section(f"min answers 2")
    binaryCorrelation(min_answers=2)
    detailedCorrelation(min_answers=2)

    table_printer.add_section(f"min answers 5")
    binaryCorrelation(min_answers=5)
    detailedCorrelation(min_answers=5)

    table_printer.export(Path(correlation_out_file))

def selfRated_vs_judged_correlation(correlation_out_file:Path, grade_filter, query_paragraphs
                                    , relevant_grade_set:Set[int],  non_relevant_grade_set:Set[int]):
    print("\n\n binary correlation")

    table_printer = print_correlation_table.TablePrinter()
    table_printer.add_section("correlation tables")
        
    for labels in [{0},{1,2,3,4,5}]:
        for judgments in [non_relevant_grade_set,relevant_grade_set]:
            print(f"\n predicted_judgment {labels} /  exact_judgment {judgments}")

            corrAll, corrPerQuery = exam_judgment_correlation.confusion_predicted_judgments_correlation(query_paragraphs, grade_filter=grade_filter, judgments=judgments, prediction=labels, min_answers=1, use_ratings=True)
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

            if 1 in non_relevant_grade_set: 
                label_to_judgment_kappa[fmt_labels({5})]=fmt_judgments({3})
                label_to_judgment_kappa[fmt_labels({4})]=fmt_judgments({3})
                label_to_judgment_kappa[fmt_labels({3})]=fmt_judgments({2})
                label_to_judgment_kappa[fmt_labels({2})]=fmt_judgments({2})                
                label_to_judgment_kappa[fmt_labels({1})]=fmt_judgments({2})                

            label_judgments_correlation_table(table_printer=table_printer
                                            , query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                            , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                            , label_to_judgment_kappa=label_to_judgment_kappa
                                            ,  judgment_title="Judgments",   label_title="GRADED"
                                            , min_answers=min_answers, use_ratings=True)
            table_printer.add_new_paragraph()
        detailedCorrelation()

    
        def mergedCorrelation():
            print("\n\n detailed correlation")
        
            predicted_label_list = [{5,4}, {3,2,1},{0}]
            judgment_list = [{3,2},{1},{0}]
            if 1 in non_relevant_grade_set:
                judgment_list = [{3},{2},non_relevant_grade_set]

            
            label_to_judgment_kappa:Dict[str, str]
            label_to_judgment_kappa = { fmt_labels(l): fmt_judgments(j) for l,j in zip( predicted_label_list, judgment_list)}


            label_judgments_correlation_table(table_printer=table_printer
                                            , query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                            , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                            , label_to_judgment_kappa=label_to_judgment_kappa
                                            ,  judgment_title="Judgments",   label_title="MERGE"
                                            , min_answers=min_answers, use_ratings=True)
            table_printer.add_new_paragraph()
        mergedCorrelation()


        def binaryLenientCorrelation():
            print("\n\n binary correlation")
        
            predicted_label_list = [{3,4,5,1,2},{0}]
            judgment_list = [relevant_grade_set, non_relevant_grade_set]

            label_to_judgment_kappa:Dict[str, str]
            label_to_judgment_kappa = { fmt_labels(l): fmt_judgments(j) for l,j in zip( predicted_label_list, judgment_list)}

            label_judgments_correlation_table(table_printer=table_printer
                                            ,query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                            , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                            , label_to_judgment_kappa=label_to_judgment_kappa
                                            ,  judgment_title="Judgments",   label_title="LENIENT"
                                            , min_answers=min_answers, use_ratings=True)
            
            table_printer.add_new_paragraph()
        binaryLenientCorrelation()




        def binaryStrictCorrelation():
            print("\n\n binary correlation")
        
            predicted_label_list = [{4,5},{3,1,2,0}]
            judgment_list = [relevant_grade_set, non_relevant_grade_set]
            
            label_to_judgment_kappa:Dict[str, str]
            label_to_judgment_kappa = { fmt_labels(l): fmt_judgments(j) for l,j in zip( predicted_label_list, judgment_list)}

            label_judgments_correlation_table(table_printer=table_printer
                                            ,query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                            , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                            , label_to_judgment_kappa=label_to_judgment_kappa
                                            ,  judgment_title="Judgments",   label_title="STRICT"
                                            , min_answers=min_answers, use_ratings=True)
            
            table_printer.add_new_paragraph()
        binaryStrictCorrelation()
    correlation_analysis(min_answers=1)
    correlation_analysis(min_answers=2)
    correlation_analysis(min_answers=5)

    table_printer.export(correlation_out_file)




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
    parser.add_argument('exam_annotated_file', type=str, metavar='exam-xxx.jsonl.gz'
                        , help='json file that annotates each paragraph with a number of answerable questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )
    parser.add_argument('-q', '--qrel-out', type=str, metavar="FILE", help='Export Qrels to this file', default=None)
    parser.add_argument('--qrel-leaderboard-out', type=str, metavar="FILE", help='Export Exam-Qrels leaderboard to this file', default=None)
    parser.add_argument('--qrel-query-facets', action='store_true', help='If set, will use query facets for qrels (prefix of question_ids)', default=None)
    parser.add_argument('--run-dir', type=str, metavar="DIR", help='Directory of trec_eval run-files. These must be uncompressed, the filename must match the pattern "${methodname}.run" where methodname refers to the method name in the official leaderboard. If set, will use the exported qrel file to determine correlation with the official leaderboard', default=None)
    parser.add_argument('--trec-eval-qrel-correlation',  type=str, metavar="IN-FILE", help='Will use this qrel file to measure leaderboard correlation with trec_eval', default=None)
    parser.add_argument('--min-trec-eval-level',  type=int, metavar="LEVEL", help='Relevance cutoff level for trec_eval. If not set, multiple levels will be tried', default=None)

    parser.add_argument('--correlation-out', type=str, metavar="FILE", help='Export Inter-annotator Agreement Correlation to this file ', default=None)

    parser.add_argument('--leaderboard-out', type=str, metavar="FILE", help='Export Leaderboard to this file ', default=None)

    parser.add_argument('-m', '--model', type=str, metavar="HF_MODEL_NAME", help='the hugging face model name used by the Q/A module.')
    parser.add_argument('--prompt-class', type=str, choices=get_prompt_classes(), required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))
    parser.add_argument('-r', '--use-ratings', action='store_true', help='If set, correlation analysis will use graded self-ratings. Default is to use the number of correct answers.')
    parser.add_argument('--min-self-rating', type=int, metavar="RATING", help='If set, will only count ratings >= RATING as relevant. (Only applies to when -r is used.)')
    parser.add_argument('--question-set', type=str, choices=["tqa","naghmeh","question-bank"], metavar="SET ", help='Which question set to use. Options: tqa or naghmeh ')
    parser.add_argument('--testset', type=str, choices=["cary3","dl19"], required=True, metavar="SET ", help='Which question set to use. Options: tqa or naghmeh ')
    parser.add_argument('--official-leaderboard', type=str, metavar="JSON-FILE", help='Use leaderboard JSON file instead (format {"methodName":rank})', default=None)
    

    # Parse the arguments
    args = parser.parse_args(args=cmdargs)    
        
    grade_filter = GradeFilter(model_name=args.model, prompt_class = args.prompt_class, is_self_rated=None, min_self_rating=None, question_set=args.question_set)


    exam_input_file=args.exam_annotated_file
    use_ratings=args.use_ratings

    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)



    official_leaderboard:Dict[str,float]
    non_relevant_grades = None
    relevant_grades = None
    if args.testset == "cary3":
        official_leaderboard = exam_leaderboard_correlation.official_CarY3_leaderboard 
    elif args.testset == "dl19":
        official_leaderboard = exam_leaderboard_correlation.official_DL19_leaderboard
        non_relevant_grades = {0,1}
        relevant_grades = {2,3}
    if args.official_leaderboard is not None:
        official_leaderboard = exam_leaderboard_correlation.load_leaderboard(args.official_leaderboard)


    if args.trec_eval_qrel_correlation is not None:
        if args.run_dir is not None:
            run_qrel_leaderboard(qrels_file=Path(args.trec_eval_qrel_correlation)
                                 ,run_dir=Path(args.run_dir)
                                 , min_level=args.min_trec_eval_level
                                 , official_leaderboard=official_leaderboard
                                 , leaderboard_out=args.qrel_leaderboard_out
                                 )


    if args.qrel_out is not None:
        export_qrels(query_paragraphs=query_paragraphs
                     , qrel_out_file=args.qrel_out
                     , grade_filter=grade_filter
                     , use_query_facets=args.qrel_query_facets
                     , use_ratings=args.use_ratings
                     )
        print("qrel leaderboard")

        if args.run_dir is not None:
            run_qrel_leaderboard(qrels_file=Path(args.qrel_out)
                                 ,run_dir=Path(args.run_dir)
                                 , min_level=args.min_trec_eval_level
                                 , official_leaderboard=official_leaderboard
                                 , leaderboard_out=args.qrel_leaderboard_out
                                 )

    if args.correlation_out is not None:
        run_interannotator_agreement(correlation_out_file=args.correlation_out
                                     , grade_filter=grade_filter
                                     , use_ratings=use_ratings
                                     , query_paragraphs=query_paragraphs
                                     , non_relevant_grades=non_relevant_grades
                                     , relevant_grades = relevant_grades
                                     )


    if args.leaderboard_out is not None:
        run_leaderboard(leaderboard_file=args.leaderboard_out
                        , grade_filter=grade_filter
                        , query_paragraphs=query_paragraphs
                        , use_ratings=use_ratings
                        , min_self_rating=args.min_self_rating
                        , official_leaderboard=official_leaderboard
                        )



if __name__ == "__main__":
    main()
