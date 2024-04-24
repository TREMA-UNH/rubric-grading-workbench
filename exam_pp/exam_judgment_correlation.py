
from typing import Set, List, Tuple
from typing import *
from pathlib import Path
import heapq

from .test_bank_prompts import *
from .data_model import FullParagraphData, Grades, QueryWithFullParagraphList, parseQueryWithFullParagraphs, GradeFilter


def frac(num:int,den:int)->float:
    return ((1.0 * num) / (1.0 * den)) if den>0 else 0.0



class ConfusionStats:
    predictedRelevant: int
    notPredictedButRelevant: int
    predictedButNotRelevant: int
    notPredictedNotRelevant: int



    def __init__(self):
        self.predictedRelevant = 0
        self.notPredictedButRelevant = 0
        self.predictedButNotRelevant = 0
        self.notPredictedNotRelevant = 0

    def __str__(self)->str:
        return f'predictedRelevant={self.predictedRelevant}  predictedButNotRelevant={self.predictedButNotRelevant}  notPredictedButRelevant={self.notPredictedButRelevant}  notPredictedNotRelevant={self.notPredictedNotRelevant}  '

    def all(self)->int:
        return self.predictedRelevant + self.predictedButNotRelevant + self.notPredictedButRelevant + self.notPredictedNotRelevant
    def empty(self)->bool:
        return self.all() <1

    def accuracy_measure(self):
        return frac(self.predictedRelevant + self.notPredictedNotRelevant,  self.all())
    def prec_measure(self):
        return frac(self.predictedRelevant, self.predictedRelevant + self.predictedButNotRelevant)
    def rec_measure(self):
        return frac(self.predictedRelevant, self.predictedRelevant + self.notPredictedButRelevant)

    def cohen_kappa(self)->float:

        total = self.all() # self.predictedRelevant + self.predictedButNotRelevant + self.notPredictedButRelevant + self.notPredictedNotRelevant
        po = frac(self.predictedRelevant + self.notPredictedNotRelevant,  total)

        pyes = frac(self.predictedRelevant + self.predictedButNotRelevant, total)
        pno = frac(self.notPredictedButRelevant + self.notPredictedNotRelevant,  total)
        pyes_rated = frac(self.predictedRelevant + self.notPredictedButRelevant,  total)
        pno_rated = frac(self.predictedButNotRelevant + self.notPredictedNotRelevant , total)

        pe = (pyes * pyes_rated) + (pno * pno_rated)
        if(pe==1.0):
            print(f'Warning, pe=1.0, can\'t compute kappa. {self}')
            return 0.0
        
        kappa = (po - pe) / (1 - pe)
        return kappa


    def printMeasures(self)->str:
        return f'kappa {self.cohen_kappa():.2f} / tp {self.predictedRelevant} ' # / acc {self.accuracy_measure():.2f} / prec {self.prec_measure():.2f} / rec {self.rec_measure():.2f}'


    def add(self, predict:bool, truth:bool):
        if (predict and truth):
            self.predictedRelevant += 1
        elif predict and not truth:
            self.predictedButNotRelevant += 1
        elif not predict and truth:
            self.notPredictedButRelevant += 1
        elif not predict and not truth:
            self.notPredictedNotRelevant += 1
        else:
            raise RuntimeError("ConfusionStats exhausted cases")



def confusion_exam_vs_judged_correlation_file(exam_input_file:Path, grade_filter:GradeFilter, min_judgment_level:int, min_answers:int=1):
    """Acc/P/R correlation between exam, judgments """
    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)

    globalExamVsJudged, perQueryStats = confusion_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, min_judgment_level=min_judgment_level, min_answers=min_answers)

    print(perQueryStats)
    print(f'all: examVsJudged {globalExamVsJudged.printMeasures()} , stats {globalExamVsJudged} ')#

def confusion_exam_vs_judged_correlation(query_paragraphs:List[QueryWithFullParagraphList], grade_filter:GradeFilter, min_judgment_level:int, min_answers:int, min_rating:Optional[int]=None)->Tuple[ConfusionStats, Dict[str, ConfusionStats]]:
    ''' workhorse to measure the per-paragraph correlation between manual judgments and exam labels.
    Only binary correlations are considered: 
        * `min_judgment_level` sets the judgment level (=>) to be considered relevant by manual judges
        * `min_answers` sets the minimum correctly answered questions (=>) to be considered relevant by EXAM

    The return value is a tuple of overall `ConfusionStats` and dictionary of per-query `ConfusionStats`
    Load files with `parseQueryWithFullParagraphs` 
    or use convenience function `confusion_exam_vs_judged_correlation_file`
    Print output with `ConfusionStats.printMeasures`, more measures are provided in `ConfusionStats`.
    '''
    globalExamVsJudged = ConfusionStats()
    perQueryStats:Dict[str,ConfusionStats] = dict()

    
    for queryWithFullParagraphList in query_paragraphs:
        examVsJudged = ConfusionStats()

        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs
        for para in paragraphs:
            judg = para.get_any_judgment()
            
            if judg ==None:
                pass # we don't have judgments
            elif not any(para.retrieve_exam_grade_any(grade_filter=grade_filter)):
                pass # we don't have labels
            else:
                # both judgments and relevance labels
                for exam_grade in para.retrieve_exam_grade_any(grade_filter=grade_filter): # there will be 1 or 0
                    hasAnsweredAny = (len(exam_grade.correctAnswered) >= min_answers)
                    if min_rating is not None:
                        if exam_grade.self_ratings is None: 
                            raise RuntimeError("These grades don't have self-ratings.")
                        filteredRatedAnswers =  [rate.get_id() for rate in exam_grade.self_ratings if rate.self_rating>=min_rating]
                        hasAnsweredAny = (len(filteredRatedAnswers) >= min_answers)
                    isJudgedRelevant = any (j.relevance>= min_judgment_level for j in para.paragraph_data.judgments)

                    globalExamVsJudged.add(predict=hasAnsweredAny, truth=isJudgedRelevant)

                    examVsJudged.add(predict=hasAnsweredAny, truth=isJudgedRelevant)

        # print(f'{query_id}: examVsJudged {examVsJudged.printMeasures()}')# ; manualRankMetric {manualRankMetric.printMeasures()}  ; examRankMetric {examRankMetric.printMeasures()}')
        perQueryStats[query_id]=examVsJudged
    return globalExamVsJudged,perQueryStats 
    # ; manualRankMetric {globalManualRankMetric.printMeasures()}  ; examRankMetric{globalExamRankMetric.printMeasures()}')


def confusion_exact_rating_exam_vs_judged_correlation(query_paragraphs:List[QueryWithFullParagraphList], grade_filter:GradeFilter, exact_judgment_level:int, min_answers:int, exact_rating:Optional[int]=None, min_rating:Optional[int]=None)->Tuple[ConfusionStats, Dict[str, ConfusionStats]]:
    ''' workhorse to measure the per-paragraph correlation between manual judgments and exam labels.
    Only binary correlations are considered: 
        * `min_judgment_level` sets the judgment level (=>) to be considered relevant by manual judges
        * `min_answers` sets the minimum correctly answered questions (=>) to be considered relevant by EXAM

    The return value is a tuple of overall `ConfusionStats` and dictionary of per-query `ConfusionStats`
    Load files with `parseQueryWithFullParagraphs` 
    or use convenience function `confusion_exam_vs_judged_correlation_file`
    Print output with `ConfusionStats.printMeasures`, more measures are provided in `ConfusionStats`.
    '''
    globalExamVsJudged = ConfusionStats()
    perQueryStats:Dict[str,ConfusionStats] = dict()

    
    for queryWithFullParagraphList in query_paragraphs:
        examVsJudged = ConfusionStats()

        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs
        for para in paragraphs:
            for exam_grade in para.retrieve_exam_grade_any(grade_filter=grade_filter): # there will be 1 or 0
                judg = para.get_any_judgment()
                
                if judg ==None:
                    continue # we don't have judgments
                if not any(para.retrieve_exam_grade_any(grade_filter=grade_filter)):
                    continue # we don't have labels
                if exam_grade.self_ratings is None:
                    raise RuntimeError(f"{query_id} paragraphId: {para.paragraph_id}:  Exam grades have no self ratings!  {exam_grade}")

                hasAnsweredAny = (len(exam_grade.correctAnswered) >= min_answers)
                if exact_rating is not None:
                    filteredRatedAnswers =  [rate.get_id() for rate in exam_grade.self_ratings if rate.self_rating==exact_rating]
                    hasAnsweredAny = (len(filteredRatedAnswers) >= min_answers)
                elif min_rating is not None:
                    filteredRatedAnswers =  [rate.get_id() for rate in exam_grade.self_ratings if rate.self_rating>=min_rating]
                    hasAnsweredAny = (len(filteredRatedAnswers) >= min_answers)
                isJudgedRelevant = any (j.relevance== exact_judgment_level for j in para.paragraph_data.judgments)

                globalExamVsJudged.add(predict=hasAnsweredAny, truth=isJudgedRelevant)

                examVsJudged.add(predict=hasAnsweredAny, truth=isJudgedRelevant)

        # print(f'{query_id}: examVsJudged {examVsJudged.printMeasures()}')# ; manualRankMetric {manualRankMetric.printMeasures()}  ; examRankMetric {examRankMetric.printMeasures()}')
        perQueryStats[query_id]=examVsJudged
    return globalExamVsJudged,perQueryStats 
    # ; manualRankMetric {globalManualRankMetric.printMeasures()}  ; examRankMetric{globalExamRankMetric.printMeasures()}')



def predict_labels_from_answers(para:FullParagraphData, grade_filter:GradeFilter, min_answers:int=1)->int:
    for exam_grade in para.retrieve_exam_grade_any(grade_filter=grade_filter): # there will be 1 or 0
        if len(exam_grade.correctAnswered) >= min_answers:
            return 1
        else:
            return 0
    return 0


def predict_labels_from_grades(para:FullParagraphData, grade_filter:GradeFilter)->int:
    if para.grades is None:
        raise RuntimeError(f"paragraph \"{para.paragraph_id}\"does not have annotated `grades`. Data: {para}")

    grade: Grades
    for grade in para.retrieve_grade_any(grade_filter=grade_filter): # there will be 1 or 0
        if grade.correctAnswered:
            return 1
        else:
            return 0
    return 0


def predict_labels_from_exam_ratings(para:FullParagraphData, grade_filter:GradeFilter, min_answers:int=1)->int:
    for exam_grade in para.retrieve_exam_grade_any(grade_filter=grade_filter): # there will be 1 or 0
        if exam_grade.self_ratings is None:
            raise RuntimeError(f"paragraphId: {para.paragraph_id}:  Exam grades have no self ratings!  {exam_grade}")

        ratings = (rate.self_rating for rate in exam_grade.self_ratings)
        best_rating:int
        if min_answers > 1:
            best_rating = min( heapq.nlargest(min_answers, ratings ))
        else:   
            best_rating = max(ratings)
            
        return best_rating
    return 0

def predict_labels_from_grade_rating(para:FullParagraphData, grade_filter:GradeFilter)->int:
    if para.grades is None:
        raise RuntimeError(f"paragraph \"{para.paragraph_id}\"does not have annotated `grades`. Data: {para}")

    grade: Grades
    for grade in para.retrieve_grade_any(grade_filter=grade_filter): # there will be 1 or 0
        if grade.self_ratings is not None:
            return grade.self_ratings
    raise RuntimeError(f"paragraph \"{para.paragraph_id}\"does not have self_ratings in \"grades\". Data: {para}")

def confusion_predicted_judgments_correlation(query_paragraphs:List[QueryWithFullParagraphList]
                                              , grade_filter:GradeFilter
                                              , judgments:Set[int]
                                              , prediction:Set[int]
                                              , use_ratings:bool
                                              , min_answers:int=1
                                              , use_exam_grades:bool = True
                                              )->Tuple[ConfusionStats, Dict[str, ConfusionStats]]:
    ''' workhorse to measure the per-paragraph correlation between manual judgments and predicted labels (based on self-rated exam grades).
    '''
    globalExamVsJudged = ConfusionStats()
    perQueryStats:Dict[str,ConfusionStats] = dict()

    
    for queryWithFullParagraphList in query_paragraphs:
        examVsJudged = ConfusionStats()

        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs
        for para in paragraphs:
            judg = para.get_any_judgment()
            
            if judg == None:
                continue # we don't have judgments
            filtered_grades = para.retrieve_exam_grade_any(grade_filter=grade_filter)
            if not any(filtered_grades):
                continue # we don't have labels

            isJudgedRelevant = any (j.relevance in judgments for j in para.paragraph_data.judgments)



            predicted_judgment:int
            if use_exam_grades:
                if use_ratings and filtered_grades[0].self_ratings:
                    predicted_judgment = predict_labels_from_exam_ratings(para=para, grade_filter=grade_filter, min_answers=min_answers)
                else:
                    predicted_judgment = predict_labels_from_answers(para=para, grade_filter=grade_filter, min_answers=min_answers)
            else: # use relevance-label prompts
                if use_ratings and filtered_grades[0].self_ratings:
                    predicted_judgment = predict_labels_from_grade_rating(para=para, grade_filter=grade_filter)
                else:
                    predicted_judgment = predict_labels_from_grades(para=para, grade_filter=grade_filter)

            predicted_relevant = (predicted_judgment in prediction)

            globalExamVsJudged.add(predict=predicted_relevant, truth=isJudgedRelevant)

            examVsJudged.add(predict=predicted_relevant, truth=isJudgedRelevant)

        # print(f'{query_id}: examVsJudged {examVsJudged.printMeasures()}')# ; manualRankMetric {manualRankMetric.printMeasures()}  ; examRankMetric {examRankMetric.printMeasures()}')
        perQueryStats[query_id]=examVsJudged
    return globalExamVsJudged,perQueryStats 
    # ; manualRankMetric {globalManualRankMetric.printMeasures()}  ; examRankMetric{globalExamRankMetric.printMeasures()}')



def main():
    import argparse

    desc = f'''Analyze how well Exam grades correlated with manual judgments on the per-paragraph level.\n
              Reporting Cohen's kappa, accuracy, predicion, recall. \n
              The input file (i.e, exam_annotated_file) has to be a *JSONL.GZ file that follows this structure: \n
              \n  
                  [query_id, [FullParagraphData]] \n
              \n
               where `FullParagraphData` meets the following structure \n
             {FullParagraphData.schema_json(indent=2)}
             '''
    
    parser = argparse.ArgumentParser(description="Analyze how well Exam grades correlated with manual judgments on the per-paragraph level."
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exam_annotated_file', type=str, metavar='exam-xxx.jsonl.gz'
                        , help='json file that annotates each paragraph with a number of anserable questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )

    parser.add_argument('-j', '--judgment-level', type=int, metavar="LEVEL", help='Minimum judgment level to count as relevant (as >=), else non-relevant', default=1)
    parser.add_argument('-a', '--min-answers', type=int, metavar="NUM", help='Minimum number of correctly answered questions per paragraph to cound as relevant for exam (as >=), else non-relevant', default=1)
    parser.add_argument('-m', '--model', type=str, metavar="HF_MODEL_NAME", help='the hugging face model name used by the Q/A module.')
    # parser.add_argument('-o', '--output', type=str, metavar="FILE", help='Output QREL file name', default='output.qrels')

    args = parser.parse_args()    
    confusion_exam_vs_judged_correlation_file(exam_input_file=args.exam_annotated_file
                                                , model_name=args.model
                                                , min_judgment_level=args.judgment_level
                                                , min_answers=args.min_answers)


if __name__ == "__main__":
    main()
