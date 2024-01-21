import sys
from typing import Dict, List, Tuple

from .exam_cover_metric import frac
from .question_types import QuestionCompleteConcisePromptWithAnswerKey2, QuestionPrompt, QuestionPromptWithChoices
from . import parse_qrels_runs_with_text as parse
from . import tqa_loader


def fix_car_query_id(input:List[Tuple[str,List[tqa_loader.Question]]]) -> List[Tuple[str,List[tqa_loader.Question]]]:
    return [ ((f'tqa2:{tqa_query_id}'), payload) for tqa_query_id, payload in input]





def main():
    questionPromptWithChoices_prompt_info =  QuestionPromptWithChoices(question_id="", question="", choices={}, correct="", correctKey=None, query_id="", facet_id=None, query_text="").prompt_info()

    # graded_query_paragraphs_file = "squad2-t5-qa-tqa-exam--benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz"
    graded_query_paragraphs_file = "t5-rating-naghmehs-tqa-rating-cc-exam-qrel-runs-result.jsonl.gz"
    fixed_graded_query_paragraphs_file = f'answercorrected-{graded_query_paragraphs_file}'
    graded_query_paragraphs = parse.parseQueryWithFullParagraphs(graded_query_paragraphs_file)

    # prompt_class = "QuestionCompleteConcisePromptWithAnswerKey"
    grade_filter = parse.GradeFilter(model_name=None, prompt_class=None, is_self_rated=False, min_self_rating=None, question_set="tqa")

    new_prompt_class = "QuestionCompleteConcisePromptWithAnswerKey2"

    query2questions_plain = fix_car_query_id(tqa_loader.load_all_tqa_questions())
    query2questions:Dict[str,Dict[str, tqa_loader.Question]]
    query2questions = {query: {q.qid:q 
                            for q in qs 
                        }
                    for query, qs in query2questions_plain 
                }


    for query_paragraphs in graded_query_paragraphs:
        query_id = query_paragraphs.queryId
        print(query_id)
        questions = query2questions[query_id]
        for paragraph in query_paragraphs.paragraphs:
            para_id = paragraph.paragraph_id

            grade:parse.ExamGrades
            for grade in paragraph.retrieve_exam_grade_all(grade_filter=grade_filter):
                correct_answered = list()
                wrong_answered = list()

                for question_id, answer in grade.answers:
                    question = questions.get(question_id, None)
                    qp = tqa_loader.question_obj_to_prompt( q=question, prompt_class=new_prompt_class)
                    if question is None:
                        raise RuntimeError(f'Cant obtain question for {grade.question_id}  (for query {query_id})')
                   
                    is_correct = qp.check_answer(answer=answer)
                    if is_correct:
                        correct_answered.append(question_id)
                    else:
                        wrong_answered.append(question_id)
                        

                grade.llm_options["answer_match"]=qp.answer_match_info()

                if grade.prompt_info is not None:
                    grade.prompt_info =  qp.prompt_info(old_prompt_info= (grade.prompt_info))
                else:
                    grade.prompt_info = qp.prompt_info(old_prompt_info= questionPromptWithChoices_prompt_info)
                # grade.prompt_info["orig_prompt_class"] = grade.prompt_info["prompt_class"]
                # grade.prompt_info["prompt_class"] = new_prompt_class

                grade.correctAnswered = correct_answered
                grade.wrongAnswered = wrong_answered
                grade.exam_ratio = frac(len(correct_answered), len(correct_answered)+len(wrong_answered))
    parse.writeQueryWithFullParagraphs(file_path=fixed_graded_query_paragraphs_file
                                       , queryWithFullParagraphList=graded_query_paragraphs)
    print(f"fixed written to {fixed_graded_query_paragraphs_file}")


def main_messaround():
# def main():
    # falseNegs = [( "the Sun", "the sun"), ("fins", "fin")]
    # x = QuestionCompleteConcisePromptWithAnswerKey2(question_id="x",query_id="x",choices= {}, correct="the sun", correctKey="", query_text="", question=None, facet_id=None)
    # sun_correct = x.check_answer("the Sun")

    # x = QuestionCompleteConcisePromptWithAnswerKey2(question_id="x",query_id="x",choices= {}, correct="fin", correctKey="", query_text="", question=None, facet_id=None)
    # fins_correct = x.check_answer("fins")

    # sys.exit()



    graded_query_paragraphs_file = "squad2-t5-qa-tqa-exam--benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz"
    fixed_graded_query_paragraphs_file = f'fixed-{graded_query_paragraphs_file}'
    graded_query_paragraphs = parse.parseQueryWithFullParagraphs(graded_query_paragraphs_file)

    prompt_class = "QuestionCompleteConcisePromptWithAnswerKey"
    grade_filter = parse.GradeFilter(model_name="google/flan-t5-large", prompt_class=None, is_self_rated=None, min_self_rating=None, question_set="tqa")

    query2questions_plain = fix_car_query_id(tqa_loader.load_all_tqa_questions())
    query2questions:Dict[str,Dict[str, tqa_loader.Question]]
    query2questions = {query: {q.qid:q 
                            for q in qs 
                        }
                    for query, qs in query2questions_plain 
                }


    for query_paragraphs in graded_query_paragraphs:
        query_id = query_paragraphs.queryId
        questions = query2questions[query_id]
        for paragraph in query_paragraphs.paragraphs:
            para_id = paragraph.paragraph_id

            grade:parse.ExamGrades
            for grade in paragraph.retrieve_exam_grade_any(grade_filter=grade_filter):
                


                for question_id, answer in grade.answers:
                    question = questions.get(question_id, None)
                    qp = tqa_loader.question_obj_to_prompt( q=question, prompt_class="QuestionCompleteConcisePromptWithAnswerKey2")
                    if question is None:
                        raise RuntimeError(f'Cant obtain question for {grade.question_id}  (for query {query_id})')
                    
                    is_correct = question_id in grade.correctAnswered
                    is_correct2 = qp.check_answer(answer=answer)
                    # if(question_id == "NDQ_000048"):
                    if(not ( question.correct == "false" or question.correct == "true")):
                        if(not is_correct == is_correct2 ):
                            if(not is_correct2):
                                print(f'{query_id}  {question_id} {para_id}|  {is_correct}->{is_correct2} |  {answer} | {question.correct} | {question.question}')
                    



if __name__ == "__main__":
    main()


    