
from collections import defaultdict
import statistics
from question_types import *
from question_types import FullParagraphData, QueryWithFullParagraphList
from parse_qrels_runs_with_text import *
from typing import Set, List, Tuple

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


def dontuse():
    def confusionMatrixCorrelation():
        """Acc/P/R correlation between exam, judgments, and rankings """
        query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs("./benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz")

        globalExamVsJudged = ConfusionStats()

        globalManualRankMetric:Dict[str, ConfusionStats] = defaultdict(ConfusionStats) 
        globalExamRankMetric:Dict[str, ConfusionStats] = defaultdict(ConfusionStats)

        
        for queryWithFullParagraphList in query_paragraphs:

            examVsJudged = ConfusionStats()
            manualRankMetric = ConfusionStats()
            examRankMetric = ConfusionStats()

            query_id = queryWithFullParagraphList.queryId
            paragraphs = queryWithFullParagraphList.paragraphs
            for para in paragraphs:
                paragraph_id = para.paragraph_id
                paragraph_txt = para.text
                exam_grade = para.get_any_exam_grade()
                judg = para.get_any_judgment()
                rank = para.get_any_ranking('ICT-DRMMTKS')

                if exam_grade==None or judg ==None:
                    continue # don't have all the data

                hasAnsweredAny = len(exam_grade.correctAnswered)>0
                isJudgedRelevant = any (j.relevance>0 for j in para.paragraph_data.judgments)
                # isInTop20 = rank is not None and rank.rank <20


                globalExamVsJudged.add(predict=hasAnsweredAny, truth=isJudgedRelevant)

                for ranks in para.paragraph_data.rankings:
                    isInTop20 = rank is not None and rank.rank <20
                    globalManualRankMetric[ranks.method].add(predict=isInTop20, truth=isJudgedRelevant)
                    globalExamRankMetric[ranks.method].add(predict=isInTop20, truth=hasAnsweredAny)



                examVsJudged.add(predict=hasAnsweredAny, truth=isJudgedRelevant)
                # manualRankMetric.add(predict=isInTop20, truth=isJudgedRelevant)
                # examRankMetric.add(predict=isInTop20, truth=hasAnsweredAny)

            print(f'{query_id}: examVsJudged {examVsJudged.printMeasures()}')# ; manualRankMetric {manualRankMetric.printMeasures()}  ; examRankMetric {examRankMetric.printMeasures()}')

        print(f'all: examVsJudged {globalExamVsJudged.printMeasures()} ')#  ; manualRankMetric {globalManualRankMetric.printMeasures()}  ; examRankMetric{globalExamRankMetric.printMeasures()}')
        for method in globalExamRankMetric.keys():
            print(f' method: {method}  exam: {globalExamRankMetric[method].printMeasures()}  manual: {globalManualRankMetric[method].printMeasures()} ')


def confusion_exam_vs_judged_correlation_file(exam_input_file:Path, min_judgment_level:int, min_answers:int=1):
    """Acc/P/R correlation between exam, judgments """
    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)

    globalExamVsJudged, perQueryStats = confusion_exam_vs_judged_correlation(query_paragraphs, min_judgment_level, min_answers)

    print(perQueryStats)
    print(f'all: examVsJudged {globalExamVsJudged.printMeasures()} , stats {globalExamVsJudged} ')#

def confusion_exam_vs_judged_correlation(query_paragraphs:List[QueryWithFullParagraphList], min_judgment_level:int, min_answers:int)->Tuple[ConfusionStats, Dict[str, ConfusionStats]]:
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
            exam_grade = para.get_any_exam_grade()
            judg = para.get_any_judgment()
            
            if exam_grade==None or judg ==None:
                continue # don't have all the data

            hasAnsweredAny = len(exam_grade.correctAnswered)>=min_answers
            isJudgedRelevant = any (j.relevance>= min_judgment_level for j in para.paragraph_data.judgments)

            globalExamVsJudged.add(predict=hasAnsweredAny, truth=isJudgedRelevant)

            examVsJudged.add(predict=hasAnsweredAny, truth=isJudgedRelevant)

        print(f'{query_id}: examVsJudged {examVsJudged.printMeasures()}')# ; manualRankMetric {manualRankMetric.printMeasures()}  ; examRankMetric {examRankMetric.printMeasures()}')
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
    # parser.add_argument('-o', '--output', type=str, metavar="FILE", help='Output QREL file name', default='output.qrels')

    args = parser.parse_args()    
    confusion_exam_vs_judged_correlation_file(exam_input_file=args.exam_annotated_file
                                     , min_judgment_level=args.judgment_level
                                     , min_answers=args.min_answers)


if __name__ == "__main__":
    main()
