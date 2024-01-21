import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

from .exam_cover_metric import frac
from .question_types import QuestionCompleteConcisePromptWithAnswerKey2, QuestionPrompt, QuestionPromptWithChoices, get_prompt_classes
from . import parse_qrels_runs_with_text as parse
from . import tqa_loader


def fix_car_query_id(input:List[Tuple[str,List[tqa_loader.Question]]]) -> List[Tuple[str,List[tqa_loader.Question]]]:
    return [ ((f'tqa2:{tqa_query_id}'), payload) for tqa_query_id, payload in input]




def reverify_answers(fixed_graded_query_paragraphs_file:Path
                     , graded_query_paragraphs:List[parse.QueryWithFullParagraphList]
                     , grade_filter:parse.GradeFilter
                     , new_prompt_class:str
                     , query2questions:Dict[str,Dict[str, tqa_loader.Question]]):
    questionPromptWithChoices_prompt_info =  QuestionPromptWithChoices(question_id="", question="", choices={}, correct="", correctKey=None, query_id="", facet_id=None, query_text="").prompt_info()

    for query_paragraphs in graded_query_paragraphs:
        query_id = query_paragraphs.queryId
        print(query_id)
        questions = query2questions.get(query_id, None)
        if questions is None:
            raise RuntimeError(f'Query_id {query_id} not found in the question set. Valid query ids are: {query2questions.keys()}')

        for paragraph in query_paragraphs.paragraphs:
            para_id = paragraph.paragraph_id

            grade:parse.ExamGrades
            for grade in paragraph.retrieve_exam_grade_all(grade_filter=grade_filter):
                correct_answered = list()
                wrong_answered = list()

                for question_id, answer in grade.answers:
                    question = questions.get(question_id, None)
                    if question is None:
                        raise RuntimeError(f'Query_id {query_id}: Cant obtain question for Question_id {question_id}. Valid question ids are: {questions.keys()}')
                    
                    qp:QuestionPrompt = tqa_loader.question_obj_to_prompt( q=question, prompt_class=new_prompt_class)
                   
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

                grade.correctAnswered = correct_answered
                grade.wrongAnswered = wrong_answered
                grade.exam_ratio = frac(len(correct_answered), len(correct_answered)+len(wrong_answered))
    parse.writeQueryWithFullParagraphs(file_path=fixed_graded_query_paragraphs_file
                                       , queryWithFullParagraphList=graded_query_paragraphs)
    print(f"fixed written to {fixed_graded_query_paragraphs_file}")


def main(cmdargs=None):



    print("EXAM Re-verify Answers Utility")
    desc = f'''EXAM Re-verify Answers Utility
             '''
    

    parser = argparse.ArgumentParser(description="EXAM pipeline"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exam_annotated_file', type=str, metavar='exam-xxx.jsonl.gz'
                        , help='json file that annotates each paragraph with a number of answerable questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )
    parser.add_argument('-o', '--out', required=True, type=str, metavar="FILE", help='File like exam-xxx.jsonl.gz to write re-verified answers to', default=None)

    parser.add_argument('-m', '--model', type=str, metavar="HF_MODEL_NAME", help='the hugging face model name used by the Q/A module.')
    parser.add_argument('--prompt-class', type=str, choices=get_prompt_classes(), required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))
    parser.add_argument('--question-set', type=str, choices=["tqa","naghmeh","question-bank"], metavar="SET ", help='Which question set to use. Options: tqa or naghmeh ')
    # parser.add_argument('--testset', type=str, choices=["cary3","dl19"], required=True, metavar="SET ", help='Which question set to use. Options: tqa or naghmeh ')
    parser.add_argument('--answer-verification-prompt', type=str, required=True, metavar="PROMPT-CLASS", help='Prompt class to use for answer re-verification')
    
    args = parser.parse_args(args=cmdargs)    

    if args.question_set != "tqa":
        raise RuntimeError("Only tqa questions can be re-verified (correct answers are required)")
        
    grade_filter = parse.GradeFilter(model_name=args.model, prompt_class = args.prompt_class, is_self_rated=None, min_self_rating=None, question_set=args.question_set)


    # graded_query_paragraphs_file = "squad2-t5-qa-tqa-exam--benchmarkY3test-exam-qrels-runs-with-text.jsonl.gz"
    graded_query_paragraphs_file = args.exam_annotated_file
    # fixed_graded_query_paragraphs_file = f'answercorrected-{graded_query_paragraphs_file}'
    fixed_graded_query_paragraphs_file = args.out
    graded_query_paragraphs = parse.parseQueryWithFullParagraphs(graded_query_paragraphs_file)


    # Load and hash questions
    query2questions_plain = fix_car_query_id(tqa_loader.load_all_tqa_questions())
    query2questions:Dict[str,Dict[str, tqa_loader.Question]]
    query2questions = {query: {q.qid:q 
                            for q in qs 
                        }
                    for query, qs in query2questions_plain 
                }


    reverify_answers(args.out, graded_query_paragraphs, grade_filter, args.answer_verification_prompt, query2questions)


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


    