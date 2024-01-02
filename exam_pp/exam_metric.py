
from collections import defaultdict
import statistics
from question_types import *
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

    def printMeasures(self)->str:
        return f'tp {self.predictedRelevant} / acc {self.accuracy_measure():.2f} / prec {self.prec_measure():.2f} / rec {self.rec_measure():.2f}'


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



def questionCoverage():
    """which method covers most questions? """
    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs("./benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz")



    ratiosPerMethod:Dict[str, List[float]] = defaultdict(list) 

    def create_set_tuple(): 
        return (set(), set())
    
    for queryWithFullParagraphList in query_paragraphs:

        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs

        def gatherCorrectVsAllAnswered(paras:List[FullParagraphData])->Tuple[ Set[str], Set[str]]:
            allCorrectlyAnswered:Set[str] = set().union(*[set(grade.correctAnswered) 
                                                    for para in paras 
                                                        for grade in para.exam_grades
                                                        ])
            allQuestions:Set[str] = set().union(*[set(grade.correctAnswered + grade.wrongAnswered) 
                                            for para in paras 
                                                for grade in para.exam_grades
                                                ])
            return (allCorrectlyAnswered, allQuestions)

        (allCorrect, allQuestions) = gatherCorrectVsAllAnswered(paragraphs)
        ratiosPerMethod["_overall"].append(frac(len(allCorrect), len(allQuestions)))

        print(f'{query_id}, overall ratio {frac(len(allCorrect), len(allQuestions))}')

        correctVsAllPerMethod:Dict[str, Tuple[ Set[str], Set[str]]] = defaultdict(create_set_tuple) 
        top20_per_method:Dict[str,List[FullParagraphData]] = defaultdict(list)
        for para in paragraphs:
                for rank in para.paragraph_data.rankings:
                    if rank.rank < 20:
                        top20_per_method[rank.method].append(para)
        correctVsAllPerMethod = {method: gatherCorrectVsAllAnswered(paras)  
                                    for method, paras in top20_per_method.items()}

        for method, (correct,answered) in correctVsAllPerMethod.items():
            ratio = frac(len(correct), len(answered))
            ratiosPerMethod[method].append(ratio)

            # print(f'{query_id}, method {method}: ratio {ratio:.2f}  correct={len(correct)}  all={len(answered)}')



    for method,ratios in ratiosPerMethod.items():
        if(len(ratios)>1):
            avgRatio =   statistics.mean(ratios)
            stdRatio =   statistics.stdev(ratios)
            # if(any([ratio <=0.0 for ratio in ratios])):
            #     print(method, ratios)
            #     pass
            # geometric_mean = statistics.geometric_mean(ratios)
            print(f'OVERALL method {method}: avg ratio {avgRatio:.2f} +/0 {stdRatio:.3f}')





def main():
    # confusionMatrixCorrelation()
    questionCoverage()



if __name__ == "__main__":
    main()
