

import csv
import sys
from typing import Set, List, Tuple, Dict, Optional, Any
from pathlib import Path
from collections import defaultdict

from . import exam_grading
from . query_loader import direct_grading_prompts, json_query_loader
from . import question_bank_loader
from . import question_loader
from . import tqa_loader


from .test_bank_prompts import DirectGradingPrompt, NuggetPrompt, Prompt, QuestionPrompt, get_prompt_classes, get_prompt_type_from_prompt_class
from .data_model import FullParagraphData, QueryWithFullParagraphList, parseQueryWithFullParagraphs, GradeFilter
from .exam_cover_metric import ExamCoverEvals, ExamCoverScorerFactory, compute_exam_cover_scores
from . import exam_to_qrels
from . import exam_leaderboard_correlation
from . import exam_judgment_correlation
from .exam_judgment_correlation import ConfusionStats
from .exam_run_trec_eval import compute_exam_qrels_scores, parse_trec_eval_variance, run_trec_eval_variance, trec_eval_leaderboard
from . import print_correlation_table
from . import exam_cover_metric

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
                                      , use_exam_grades:bool
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
                                                                                                        , use_ratings=use_ratings
                                                                                                        , use_exam_grades=use_exam_grades)
            print(f'Min Answers= {min_answers}, use ratings= {use_ratings}, Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')
            counts[fmt_labels(label)][fmt_judgments(judgment)]=corrAll.predictedRelevant
            if label_to_judgment_kappa[fmt_labels(label)] == fmt_judgments(judgment):
                kappas[fmt_labels(label)] = corrAll.cohen_kappa()

    table_printer.add_table(counts=counts, kappa=kappas
                                        , judgments_header=judgments_header, label_header=label_header
                                        , judgment_title=judgment_title, label_title=label_title
                                        , label_to_judgment_kappa=label_to_judgment_kappa)
        

def export_qrels(query_paragraphs,  qrel_out_file:Path, grade_filter:GradeFilter, use_query_facets:bool = False, direct_grading:bool = False, use_ratings:bool = False, query_facets: Dict[str,Set[str]]=dict()):
    if use_query_facets:
        if direct_grading:
            if use_ratings:
                qrel_entries = exam_to_qrels.convert_direct_to_rated_facet_qrels(query_paragraphs,grade_filter=grade_filter, query_facets=query_facets)
            else:
                qrel_entries = exam_to_qrels.convert_direct_to_facet_qrels(query_paragraphs,grade_filter=grade_filter, query_facets=query_facets)

        else:
            if use_ratings:
                qrel_entries = exam_to_qrels.convert_exam_to_rated_facet_qrels(query_paragraphs,grade_filter=grade_filter, query_facets=query_facets)
            else:
                qrel_entries = exam_to_qrels.convert_exam_to_facet_qrels(query_paragraphs,grade_filter=grade_filter, query_facets=query_facets)
    else:
        if use_ratings:
            qrel_entries = exam_to_qrels.convert_exam_to_rated_qrels(query_paragraphs,grade_filter=grade_filter)
        else:
            qrel_entries = exam_to_qrels.convert_exam_to_qrels(query_paragraphs,grade_filter=grade_filter)

    exam_to_qrels.write_qrel_file(qrel_out_file, qrel_entries)
    print(f"Direct? {direct_grading}/ ratings? {use_ratings} / query-facets? {use_query_facets}: Exporting {len(qrel_entries)} to {qrel_out_file}. Grade_filter = {grade_filter.print_name()}")



def run_interannotator_agreement(correlation_out_file:Path, grade_filter, use_ratings, query_paragraphs, use_exam_grades:bool
                                 , relevant_grades:Optional[Set[int]]=None,  non_relevant_grades:Optional[Set[int]]=None):
    corrAll:ConfusionStats
    corrPerQuery:Dict[str, ConfusionStats]

    print("ignoring rating levels")

    non_relevant_grade_set = {0}
    relevant_grade_set = {1,2,3,4,5}

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
        selfRated_vs_judged_correlation(correlation_out_file, grade_filter, query_paragraphs, relevant_grade_set, non_relevant_grade_set, use_exam_grades=use_exam_grades)
    else:
        binary_vs_judged_correlation(correlation_out_file, grade_filter, query_paragraphs, relevant_grade_set, non_relevant_grade_set, use_exam_grades=use_exam_grades)

    print("\n\n exam_vs_judged")

    corrAll, corrPerQuery = exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, min_judgment_level=1, min_answers=1)
    for query_id, corr in corrPerQuery.items():
        print(f'{query_id}: examVsJudged {corr.printMeasures()}')# ; manualRankMetric {manualRankMetric.printMeasures()}  ; examRankMetric {examRankMetric.printMeasures()}')
    print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')




def cover_leaderboard_analysis(grade_filter_list:List[GradeFilter], query_paragraphs:List[QueryWithFullParagraphList], official_leaderboard:Dict[str,int], analysis_out:Path, min_levels: List[int]):

    def f2s(x:Optional[float])->str:
        if x is None:
            return ' '
        else:
            return f'{x:.3f}'
    


    with open(analysis_out, 'wt') as file:
        file.write("Leaderboard Rank Correlation Analysis Exam-Cover\n")
        # file.write(f"run_dir\t{run_dir}\n")


        file.write('\t'.join(["method"
                            , "min_rating"
                            , "trec_eval_metric"
                            , "spearman"
                            , "kendall"
                    ]))
        file.write('\n')

        for grade_filter in grade_filter_list:
            for min_level in min_levels:
                exam_factory = ExamCoverScorerFactory(grade_filter=grade_filter, min_self_rating=min_level)
                resultsPerMethod:Dict[str, ExamCoverEvals] = compute_exam_cover_scores(query_paragraphs, exam_factory=exam_factory)
                resultsPerMethod__ = [val for key, val in resultsPerMethod.items() if key != exam_cover_metric.OVERALL_ENTRY]
                # exam_leaderboard_correlation.print_leaderboard_eval(list(resultsPerMethod__.values()), grade_filter=grade_filter)

                nExamCorrelation,correlationStats=exam_leaderboard_correlation.leaderboard_correlation(resultsPerMethod__, official_leaderboard=official_leaderboard)
                pretty_min_level = f"{min_level:.0f}" if min_level else "bool"

                file.write('\t'.join([grade_filter.print_name()
                                    , f"{pretty_min_level}"
                                    , "cover"
                                    , f2s(correlationStats.spearman_correlation)
                                    , f2s(correlationStats.kendall_correlation)
                                    ]))
                file.write('\n')


        file.writelines(["\n","\n"])
        file.close()
    print(f"exam-cover leaderboard written to {analysis_out}")




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



def run_qrel_variance_leaderboard(qrels_file:Path, run_dir:Path, leaderboard_out:Path, min_level = Optional[int], trec_eval_metric:str = "P.20", grade_filter:GradeFilter=GradeFilter.noFilter(), official_leaderboard:Dict[str,float] =dict(), leaderboard_sort:Optional[str]=None):
    with open(leaderboard_out, 'wt') as file:
        # min_rating:Optional[int]

        for min_level_x in ([1,2,3,4,5] if min_level is None else [min_level]):

            resultsPerMethod:Dict[str, ExamCoverEvals]
            
            trec_eval_out =run_trec_eval_variance(run_dir=run_dir, qrels=qrels_file, min_level=min_level_x, trec_eval_metric=trec_eval_metric)
            trec_eval_parsed=parse_trec_eval_variance(trec_eval_out)
            resultsPerMethod = compute_exam_qrels_scores(trec_eval_parsed)


            
            # resultsPerMethod__ = [val for key, val in resultsPerMethod.items() if key != exam_cover_metric.OVERALL_ENTRY]
            exam_leaderboard_correlation.print_leaderboard_eval(evals = list(resultsPerMethod.values()), grade_filter=grade_filter)

            # nExamCorrelation,examCorrelation=exam_leaderboard_correlation.leaderboard_correlation(resultsPerMethod.values(), official_leaderboard=official_leaderboard)
            # print(f'min_rating={str(min_rating)} nExam:{nExamCorrelation}')
            # print(f'min_rating={str(min_rating)}  exam:{examCorrelation}')


            table = exam_leaderboard_correlation.leaderboard_table(list(resultsPerMethod.values())
                                                                   , official_leaderboard=official_leaderboard
                                                                   , nExamCorrelation=None
                                                                   , examCorrelation=None
                                                                   , sortBy=leaderboard_sort) 
            

        
            file.writelines("\n".join(table))
            file.writelines( ["\n"
                            , f' EXAM scores produced with {grade_filter}\n'
                            , f' min_rating\t{str(min_level_x)}\n'
                            ,'\n'])

            file.writelines(["\n","\n"])

        file.close()


def run_minimal_qrel_leaderboard(qrels_file:Path, run_dir:Path, leaderboard_out:Path, min_level = Optional[int], trec_eval_metric:str = "P.20"):
    with open(leaderboard_out, 'wt') as file:
        file.write("based on Exam-Qrels\n")


        print(f'run_dir={run_dir}\n qrels_file={qrels_file}\nmin_level={min_level}')
        methodScores = trec_eval_leaderboard(run_dir=run_dir, qrels=qrels_file, min_level=min_level, trec_eval_metric=trec_eval_metric)

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
        lines = exam_leaderboard_correlation.leaderboard_qrel(evals =examScores, header=f"exam-qrel-{trec_eval_metric}")
        file.write(lines+'\n')
        file.write('\n'.join([""
                            ,(f"min_rating\t{min_level:.0f}" if min_level is not None else "\t")
                            , f"qrel_file\t{qrels_file}"
                            , "\n"]))
    
        print(f'min_level\t{min_level}\n')


        file.writelines(["\n","\n"])
        file.close()
    print(f"exam-qrels leaderboard written to {leaderboard_out}")


def qrel_leaderboard_analysis(qrels_files:List[Path], run_dir:Path,  official_leaderboard:Dict[str,float], analysis_out:Path, min_levels: List[int], trec_eval_metrics:List[str]):

    def f2s(x:Optional[float])->str:
        if x is None:
            return ' '
        else:
            return f'{x:.3f}':447
    
    
    print(f'run_dir={run_dir}')

    with open(analysis_out, 'wt') as file:
        file.write("Leaderboard Rank Correlation Analysis Exam-Qrels\n")
        file.write(f"run_dir\t{run_dir}\n")


        file.write('\t'.join(["method"
                            , "min_rating"
                            , "trec_eval_metric"
                            , "spearman"
                            , "kendall"
                    ]))
        file.write('\n')

        for trec_eval_metric in trec_eval_metrics:
            for qrels_file in qrels_files:
                for min_level in min_levels:
                    methodScores = trec_eval_leaderboard(run_dir=run_dir, qrels=qrels_file, min_level=min_level, trec_eval_metric=trec_eval_metric)
                    correlationStats=exam_leaderboard_correlation.leaderboard_rank_correlation(methodScores, official_leaderboard=official_leaderboard)

                    file.write('\t'.join([f"{qrels_file}"
                                        , f"{min_level:.0f}"
                                        , trec_eval_metric
                                        , f2s(correlationStats.spearman_correlation)
                                        , f2s(correlationStats.kendall_correlation)
                                        ]))
                    file.write('\n')


        file.writelines(["\n","\n"])
        file.close()
    print(f"exam-qrels leaderboard written to {analysis_out}")



def run_qrel_leaderboard(qrels_file:Path, run_dir:Path,  official_leaderboard:Dict[str,int], leaderboard_out:Path, min_level: Optional[int], trec_eval_metric:str ="P.20"):
    with open(leaderboard_out, 'wt') as file:
        file.write("based on Exam-Qrels\n")

        for min_level_x in ([1,2,3,4,5] if min_level is None else [min_level]):

            print(f'run_dir={run_dir}\n qrels_file={qrels_file}\nmin_level={min_level_x}')
            methodScores = trec_eval_leaderboard(run_dir=run_dir, qrels=qrels_file, min_level=min_level_x, trec_eval_metric=trec_eval_metric)
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

        file.writelines(["\n","\n"])
        file.close()
    print(f"exam-qrels leaderboard written to {leaderboard_out}")

def binary_vs_judged_correlation(correlation_out_file:Path, grade_filter:GradeFilter, query_paragraphs
                                 , relevant_grade_set:Set[int],  non_relevant_grade_set:Set[int], use_exam_grades:bool):
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
                                        , min_answers=min_answers, use_ratings=False, use_exam_grades=use_exam_grades)
        
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
                                        , min_answers=min_answers, use_ratings=False, use_exam_grades=use_exam_grades)
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
                                    , relevant_grade_set:Set[int],  non_relevant_grade_set:Set[int], use_exam_grades:bool):
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
                                            , min_answers=min_answers, use_ratings=True, use_exam_grades=use_exam_grades)
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
                                            , min_answers=min_answers, use_ratings=True, use_exam_grades=use_exam_grades)
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
                                            , min_answers=min_answers, use_ratings=True, use_exam_grades=use_exam_grades)
            
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
                                            , min_answers=min_answers, use_ratings=True, use_exam_grades=use_exam_grades)
            
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
              The input file (i.e, exam_annotated_file) has to be a *JSONL.GZ file more info with --help-schema
             '''
    
    help_schema=f'''The input and output file (i.e, exam_annotated_file) has to be a *JSONL.GZ file that follows this structure: \n
                \n  
                    [query_id, [FullParagraphData]] \n
                \n
                where `FullParagraphData` meets the following structure \n
                {FullParagraphData.schema_json(indent=2)}
                \n
                Create a compatible file with 
                exam_pp.data_model.writeQueryWithFullParagraphs(file_path:Path, queryWithFullParagraphList:List[QueryWithFullParagraphList])
                '''

    parser = argparse.ArgumentParser(description="EXAM pipeline"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exam_annotated_file', type=str, metavar='exam-xxx.jsonl.gz'
                        , help='json file that annotates each paragraph with a number of answerable questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )
    parser.add_argument('-q', '--qrel-out', type=str, metavar="FILE", help='Export Qrels to this file', default=None)
    parser.add_argument('--qrel-leaderboard-out', type=str, metavar="FILE", help='Export Exam-Qrels leaderboard to this file. (applies only to -q)', default=None)
    parser.add_argument('--qrel-query-facets', action='store_true', help='If set, will use query facets for qrels (prefix of question_ids). (applies only to -q)', default=None)
    parser.add_argument('--run-dir', type=str, metavar="DIR", help='Directory of trec_eval run-files. These must be uncompressed, the filename must match the pattern "${methodname}.run" where methodname refers to the method name in the official leaderboard. If set, will use the exported qrel file to determine correlation with the official leaderboard. (applies only to -q)', default=None)
    parser.add_argument('--trec-eval-qrel-correlation',  type=str, metavar="IN-FILE", help='Will use this qrel file to measure leaderboard correlation with trec_eval (applies only to -q)', default=None)
    parser.add_argument('--min-trec-eval-level',  type=int, metavar="LEVEL", help='Obsolete. Use --min_relevant-judgment instead.  Relevance cutoff level for trec_eval. If not set, multiple levels will be tried (applies only to -q)', default=None)
    parser.add_argument('--trec-eval-metric', type=str, metavar="str", help='Which evaluation metric to use in trec_eval. Default: P.20. (applies only to -q)', default="P.20")

    parser.add_argument('--leaderboard-out', type=str, metavar="FILE", help='Export Cover Leaderboard to this file (alternative to -q)', default=None)
    parser.add_argument('-s', '--leaderboard-sort', type=str, metavar="SORT", help='Key to sort the leaderboard (exam, n-exam) or None for sort by method name.')


    parser.add_argument('--correlation-out', type=str, metavar="FILE", help='Deprecated option, use --inter-annotator-out instead!', default=None)
    parser.add_argument('--inter-annotator-out', type=str, metavar="FILE", help='Export Inter-annotator Agreement Correlation to this file ', default=None)


    parser.add_argument('-m', '--model', type=str, metavar="HF_MODEL_NAME", help='the hugging face model name used by the Q/A module.')
    parser.add_argument('--prompt-class', type=str,  required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class or custom prompt name implementation to use. Choices: "+", ".join(get_prompt_classes()))
    parser.add_argument('-r', '--use-ratings', action='store_true', help='If set, correlation analysis will use graded self-ratings. Default is to use the number of correct answers.')
    parser.add_argument('--use-relevance-prompt', action='store_true', help='If set, use relevance prompt instead of exam grades. (Inter-annotator only)')
    parser.add_argument('--min-self-rating', type=int, metavar="RATING", help='If set, will only count ratings >= RATING as relevant for leaderboards. (Only applies to when -r is used.)')
    
    parser.add_argument('--question-set', type=str, choices=["tqa","genq","question-bank"], metavar="SET ", help='Which question set to use. Options: tqa, genq,  or question-bank ')
    parser.add_argument('--question-set-for-facets', type=str, choices=["tqa","genq","question-bank"], metavar="SET ", help='Which question set to use. Options: tqa, genq,  or question-bank ')
    parser.add_argument('--question-path-for-facets', type=str, metavar='PATH', help='Path to read exam questions from (can be tqa directory or question-bank file) -- only needed for direct grading with facets')

    parser.add_argument('--official-leaderboard', type=str, metavar="JSON-FILE", help='Use leaderboard JSON file instead (format {"methodName":rank})', default=None)
    parser.add_argument('--min-relevant-judgment', type=int, default=1, metavar="LEVEL", help='Minimum judgment levelfor relevant passages. (Set to 2 for TREC DL)')
    parser.add_argument('--testset', type=str, choices=["cary3","dl19","dl20"], metavar="SET", help='Offers hard-coded defaults for --official-leaderboard and --min-relevant-judgment for some test sets. Options: cary3, dl19, or dl20 ')

    parser.add_argument('--dont-check-prompt-class',action='store_true',  help='If set, will allow any prompt_class to be used that is in the data, without any verification. Any data errors are your own fault!')
    prompt_type_choices=[QuestionPrompt.my_prompt_type, NuggetPrompt.my_prompt_type, DirectGradingPrompt.my_prompt_type]
    parser.add_argument('--prompt-type', type=str, choices=prompt_type_choices, required=False,  metavar="PROMPT_TYPE", help=f"Manually set the prompt_type when setting --dont-check-prompt-class (it will otherwise be automatically set based on known prompt_classes). Choices: {prompt_type_choices}")

    parser.add_argument('--help-schema', action='store_true', help="Additional info on required JSON.GZ input format")


    # Parse the arguments
    if cmdargs is not None:
        args = parser.parse_args(args=cmdargs)    
    else:
        args = parser.parse_args()



    if args.help_schema:
        print(help_schema)
        sys.exit()



    if not args.dont_check_prompt_class:
        if args.prompt_class not in get_prompt_classes():
            raise RuntimeError(f"Unknown promptclass {args.prompt_class}. Valid choices: {get_prompt_classes()}. You can disable the check with \'--dont-check-prompt-class\'")
    if args.dont_check_prompt_class:
        #  make sure that prompt_type argument is specified
        if args.prompt_type is None:
            raise RuntimeError(f"Since --dont-check-prompt-class is set, you must also specify --prompt-type PROMPT_TYPE!")

    def get_prompt_type():
        if args.dont_check_prompt_class:
            return args.prompt_type
        else:
            return get_prompt_type_from_prompt_class(args.prompt_class)

    grade_filter = GradeFilter(model_name=args.model, prompt_class = args.prompt_class, is_self_rated=None, min_self_rating=None, question_set=args.question_set, prompt_type=get_prompt_type())


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


    query_facets:Dict[str,Set[str]] = {}
    if args.qrel_query_facets: # and get_prompt_type_from_prompt_class(args.prompt_class)==DirectGradingPrompt.my_prompt_type:
        # we have to load the questions and get facets for each query
        # so we can emit facet-based query information with the qrel file

        print("Loading query facets for direct grading qrels")
        question_set_for_facets = args.question_set_for_facets if args.question_set_for_facets  else args.question_set

        if question_set_for_facets == "tqa":
            tqabank = tqa_loader.load_TQA_questions(tqa_file=args.question_path_for_facets)
            for query_id, tqa_questions in tqabank:
                if not query_id in query_facets:
                    query_facets[query_id]=set()
                for tqa_question in tqa_questions:
                    query_facets[query_id].add(tqa_question.facet_id)

        elif question_set_for_facets == 'question-bank':
            testbank = question_bank_loader.parseTestBank(file_path=args.question_path_for_facets, use_nuggets=False)
            for bank in testbank:
                if not bank.query_id in query_facets:
                    query_facets[bank.query_id]=set()
                query_facets[bank.query_id].add(bank.facet_id)

        else:
            raise RuntimeError(f"loading of facets for question set {question_set_for_facets} from path {args.question_path_for_facets} is not implemented")




    if args.trec_eval_qrel_correlation is not None:
        if args.run_dir is not None:
            # run_qrel_leaderboard(qrels_file=Path(args.trec_eval_qrel_correlation)
            #                      ,run_dir=Path(args.run_dir)
            #                      , min_level=args.min_trec_eval_level or args.min_self_rating
            #                      , official_leaderboard=official_leaderboard
            #                      , leaderboard_out=args.qrel_leaderboard_out
            #                      , trec_eval_metric=args.trec_eval_metric
            #                      )

            run_qrel_variance_leaderboard(qrels_file=Path(args.trec_eval_qrel_correlation)
                                 ,run_dir=Path(args.run_dir)
                                 , min_level=args.min_trec_eval_level or args.min_self_rating
                                 , leaderboard_out=args.qrel_leaderboard_out
                                 , trec_eval_metric=args.trec_eval_metric
                                 , leaderboard_sort=args.leaderboard_sort
                                 , official_leaderboard=official_leaderboard
                                 , grade_filter=grade_filter
                                 )


    if args.qrel_out is not None:
        export_qrels(query_paragraphs=query_paragraphs
                     , qrel_out_file=args.qrel_out
                     , grade_filter=grade_filter
                     , use_query_facets=args.qrel_query_facets
                     , use_ratings=args.use_ratings
                     , query_facets=query_facets
                     , direct_grading= get_prompt_type()==DirectGradingPrompt.my_prompt_type
                    #  , direct_grading= args.qrel_query_facets and get_prompt_type()==DirectGradingPrompt.my_prompt_type
                     )
        print("qrel leaderboard")

        if args.run_dir is not None:
            # run_qrel_leaderboard(qrels_file=Path(args.qrel_out)
            #                      ,run_dir=Path(args.run_dir)
            #                      , min_level=args.min_trec_eval_level or args.min_self_rating
            #                      , official_leaderboard=official_leaderboard
            #                      , leaderboard_out=args.qrel_leaderboard_out
            #                      , trec_eval_metric=args.trec_eval_metric
            #                     # , grade_filter = grade_filter
            #                      )

            run_qrel_variance_leaderboard(qrels_file=Path(args.qrel_out)
                                 ,run_dir=Path(args.run_dir)
                                 , min_level=args.min_trec_eval_level or args.min_self_rating
                                 , leaderboard_out=args.qrel_leaderboard_out
                                 , trec_eval_metric=args.trec_eval_metric
                                 , leaderboard_sort=args.leaderboard_sort
                                 , official_leaderboard=official_leaderboard
                                 , grade_filter=grade_filter
                                 )


    if args.inter_annotator_out is not None or args.correlation_out is not None:
        correlation_out = args.inter_annotator_out if args.inter_annotator_out is not None else args.correlation_out
        run_interannotator_agreement(correlation_out_file=correlation_out
                                     , grade_filter=grade_filter
                                     , use_ratings=use_ratings
                                     , query_paragraphs=query_paragraphs
                                     , non_relevant_grades=non_relevant_grades
                                     , relevant_grades = relevant_grades
                                     , use_exam_grades = not (args.use_relevance_prompt)
                                     )


    if args.leaderboard_out is not None:
        # run_qrel_variance_leaderboard
        run_leaderboard(leaderboard_file=args.leaderboard_out
                        , grade_filter=grade_filter
                        , query_paragraphs=query_paragraphs
                        , use_ratings=use_ratings
                        , min_self_rating=args.min_self_rating
                        , official_leaderboard=official_leaderboard
                        )



if __name__ == "__main__":
    main()
