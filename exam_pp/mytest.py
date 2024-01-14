

from exam_judgment_correlation import ConfusionStats
import exam_judgment_correlation
from parse_qrels_runs_with_text import GradeFilter, QueryWithFullParagraphList, parseQueryWithFullParagraphs
from typing import *

def main_other():
    prompt_class = "QuestionCompleteConcisePromptWithAnswerKey"
    model = "google/flan-t5-large"
    exam_input_file = "./t5-cc-rating-exam-qrel-result.jsonl.gz"


    grade_filter = GradeFilter(model_name=model, prompt_class = prompt_class, is_self_rated=None, min_self_rating=None, question_set="tqa")

    print('other', grade_filter)

    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)
    corrAll:ConfusionStats
    corrPerQuery:Dict[str, ConfusionStats]



    for queryWithFullParagraphList in query_paragraphs:
        examVsJudged = ConfusionStats()

        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs
        # for para in paragraphs:
        #     if len(para.retrieve_exam_grade(grade_filter=grade_filter))>1:
        #            raise RuntimeError('there should only be 0 or 1')
            
        #     for exam_grade in para.retrieve_exam_grade(grade_filter=grade_filter.get_min_grade_filter(4)): # there will be 1 or 0

        #         ratings = DefaultDict(list)
        #         for rate in exam_grade.self_ratings:
        #             ratings[rate.self_rating].append(rate.question_id)
                
        #         rate_stats = {rating: len(qlist) for rating, qlist in ratings.items()}

        #         if rate_stats.get(4,None) is None and rate_stats.get(5,None) is None:
        #             print(query_id, para.paragraph_id, exam_grade)
        #             pass

        #         print(query_id, para.paragraph_id, '4:',rate_stats.get(4), '5:',rate_stats.get(5))



    corrAll, corrPerQuery = exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, min_judgment_level=1, min_answers=1)

    print(corrAll, 'all:', corrAll.all())
    print(corrAll.printMeasures())




def main_rating():
    prompt_class = "QuestionSelfRatedUnanswerablePromptWithChoices"
    model = "google/flan-t5-large"
    exam_input_file = "./t5-cc-rating-exam-qrel-result.jsonl.gz"


    grade_filter = GradeFilter(model_name=model, prompt_class = prompt_class, is_self_rated=None, min_self_rating=None, question_set="tqa")

    print('rating', grade_filter)

    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)
    corrAll:ConfusionStats
    corrPerQuery:Dict[str, ConfusionStats]



    for queryWithFullParagraphList in query_paragraphs:
        examVsJudged = ConfusionStats()

        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs
        for para in paragraphs:
            if len(para.retrieve_exam_grade(grade_filter=grade_filter))>1:
                   raise RuntimeError('there should only be 0 or 1')
            
            # for exam_grade in para.retrieve_exam_grade(grade_filter=grade_filter.get_min_grade_filter(4)): # there will be 1 or 0

            #     ratings = DefaultDict(list)
            #     for rate in exam_grade.self_ratings:
            #         ratings[rate.self_rating].append(rate.question_id)
                
            #     rate_stats = {rating: len(qlist) for rating, qlist in ratings.items()}

            #     if rate_stats.get(4,None) is None and rate_stats.get(5,None) is None:
            #         print(query_id, para.paragraph_id, exam_grade)
            #         pass

            #     print(query_id, para.paragraph_id, '4:',rate_stats.get(4), '5:',rate_stats.get(5))



    min_rating_level = 1
    print(f'min_rating={min_rating_level}')
    corrAll, corrPerQuery = exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, min_judgment_level=1, min_answers=1, min_rating=min_rating_level)

    print(corrAll, 'all:', corrAll.all())
    print(corrAll.printMeasures())



if __name__ == "__main__":
    main_rating()
    # main_other()
