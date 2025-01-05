import argparse
from collections import defaultdict
from dataclasses import dataclass
import gzip
import itertools
import os
from pathlib import Path
from random import shuffle
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from transformers import pipeline, T5ForConditionalGeneration, AutoTokenizer


from .exam_cover_metric import frac
from .test_bank_prompts import QuestionCompleteConcisePromptWithAnswerKey2, QuestionPrompt, QuestionPromptWithChoices, get_prompt_classes, get_prompt_type_from_prompt_class
from . import data_model as parse
from . import tqa_loader




os.environ["DSP_NOTEBOOK_CACHEDIR"] = str((Path(".") / "cache").resolve())
device:Optional[int] = None
deviceStr = os.environ.get("GPU_DEVICE")
if deviceStr is not None:
    try:
        device = int(deviceStr)
    except ValueError:
        print(f'Cant parse device number from \"{device}\"')
        device = None

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
MAX_TOKEN_LEN = 512
print(f'Device = {device}; BATCH_SIZE = {BATCH_SIZE}')




def fix_car_query_id(input:List[Tuple[str,List[tqa_loader.Question]]]) -> List[Tuple[str,List[tqa_loader.Question]]]:
    return [ ((f'tqa2:{tqa_query_id}'), payload) for tqa_query_id, payload in input]




def reverify_answers(fixed_graded_query_paragraphs_file:Path
                     , graded_query_paragraphs:List[parse.QueryWithFullParagraphList]
                     , grade_filter:parse.GradeFilter
                     , new_prompt_class:str
                     , query2questions:Dict[str,Dict[str, tqa_loader.Question]]):
    questionPromptWithChoices_prompt_info =  QuestionPromptWithChoices(question_id="", question="", choices={}, correct="", correctKey=None, query_id="", facet_id=None, query_text="").prompt_info()

    with gzip.open(fixed_graded_query_paragraphs_file, 'wt', encoding='utf-8') as file:

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

            file.write(parse.dumpQueryWithFullParagraphList(query_paragraphs))
            file.flush()
            pass

        file.close()
        # parse.writeQueryWithFullParagraphs(file_path=fixed_graded_query_paragraphs_file
        #                                    , queryWithFullParagraphList=graded_query_paragraphs)
        print(f"fixed written to {fixed_graded_query_paragraphs_file}")




@dataclass
class AnswerVerificationData():
    question:tqa_loader.Question
    answer:str
    llm_response:Optional[str]
    is_correct:Optional[bool]

    def add_response(self, response):
        self.llm_response = response

        prompt = self.get_prompt()
        question =  self.question.question
        correct_answer = self.question.correct
        answer = self.answer


        if response.strip() == "yes":
            self.is_correct = True
        if response.strip() == "no":
            self.is_correct = False

        print(f"{self.question.query_id}/{self.question.qid}| {response.strip()} | {answer} | {correct_answer} | {question} ")
        if self.is_correct == None:
            print(f'{self.question.query_id} Cant verify as response is \"{response}\":  {response.strip()} | answer: {answer} | correct: {correct_answer} | question: {question} ')
            pass
        if self.is_correct:
            pass
        pass

    def get_prompt(self)->str:
        question =  self.question.question
        correct_answer = self.question.correct
        answer = self.answer

        if (self.question.correct == "all of the above")  and (self.question.choices is not None):
            correct_answer = "all of "+ (" and ".join(self.question.choices.values()))
        if (self.question.correct == "none of the above") and (self.question.choices is not None):
            correct_answer = "neither "+ (" nor ".join(self.question.choices.values()))
        return f"For the question \"{question}\" the correct answer is \"{correct_answer}\". Is \"{answer}\" an equally correct response to this question? Answer yes or no."

        # return f"Please verify if the answer \"{answer}\" is a correct response to this question"

class Text2TextPipeline():
    """QA Pipeline for text2text based question answer checking"""

    def __init__(self, model_name:str):
        self.question_batchSize = 100 # batchSize
    
        # Initialize the tokenizer and model
        # self.modelName = 'google/flan-t5-large'
        self.modelName = model_name
        self.model = T5ForConditionalGeneration.from_pretrained(self.modelName)
        # self.tokenizer = T5TokenizerFast.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)

        print(f"Text2Text model config: { self.model.config}")
        # print("maxBatchSize",computeMaxBatchSize(self.model.config))
        # self.promptGenerator = promptGenerator
        self.max_token_len = 512

        # Create a Hugging Face pipeline
        self.t5_pipeline_qa = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer, device=device, batch_size=BATCH_SIZE, use_fast=True)

    def exp_modelName(self)->str:
        return self.modelName


    def batchChunker(self, iterable):
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, self.question_batchSize))
            if not batch or len(batch)<1:
                break
            yield batch


    def verify_answers_with_t5(self, verification_data:List[AnswerVerificationData])->List[AnswerVerificationData]:

            def processBatch(verification_data:List[AnswerVerificationData])->Iterable[AnswerVerificationData]:
                """Prepare a batch of prompts, tuple it up with the answers"""
                prompts = [d.get_prompt() for d in verification_data ]
                outputs = self.t5_pipeline_qa(prompts, max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
                responses:List[str] = [output['generated_text']  for output in outputs]

                for d,r in zip(verification_data, responses, strict=True):
                    d.add_response(r)
                return verification_data

            return list(itertools.chain.from_iterable(
                        (processBatch(batch) for batch in self.batchChunker(verification_data)) 
                        )) 



class AnswerCollection():
    def __init__(self):
        ## Define work horse data structure
        # collected_answers_: Dict[str, Dict[str, List[parse.FullParagraphData]]] = defaultdict(dict)
        self.collected_answers: Dict[str, Dict[str, Optional[AnswerVerificationData]]] 
        self.collected_answers = defaultdict(dict)


    # {question_id : { answer: [paragraphs] }}
    def add_answer(self, question_id:str, answer:str, para:parse.FullParagraphData):
        self.collected_answers[question_id][answer.strip()] = None
    
    def get_verified_answer(self, question_id:str, answer:str, para:parse.FullParagraphData)->Optional[bool]:
        answer_verification_maybe:Optional[AnswerVerificationData]
        answer_verification_maybe = self.collected_answers.get(question_id, {}).get(answer.strip(), None)
        if answer_verification_maybe is None:
            return None
        else:
            return answer_verification_maybe.is_correct


    def answers_to_check(self, question_id:str)->Iterable[str]:
        return self.collected_answers[question_id].keys()


    def set_verification_data(self, question_id:str, answer:str, verification_data:AnswerVerificationData):
        self.collected_answers[question_id][answer.strip()]=verification_data

    def all_verification_data(self):
        return [verification_data for xs in self.collected_answers.values() for verification_data in xs.values()]


def t5_verify_answers(fixed_graded_query_paragraphs_file:Path
                     , graded_query_paragraphs:List[parse.QueryWithFullParagraphList]
                     , grade_filter:parse.GradeFilter
                     , new_prompt_class:str
                     , query2questions:Dict[str,Dict[str, tqa_loader.Question]], sample:Optional[int]=None):
    questionPromptWithChoices_prompt_info =  QuestionPromptWithChoices(question_id="", question="", choices={}, correct="", correctKey=None, query_id="", facet_id=None, query_text="").prompt_info()

    grade:parse.ExamGrades
    t5 = Text2TextPipeline("google/flan-t5-large")

    with gzip.open(fixed_graded_query_paragraphs_file, 'wt', encoding='utf-8') as file:

        for query_paragraphs in graded_query_paragraphs:
            query_id = query_paragraphs.queryId
            print(query_id)
            questions = query2questions.get(query_id, None)
            if questions is None:
                raise RuntimeError(f'Query_id {query_id} not found in the question set. Valid query ids are: {query2questions.keys()}')

            collected_answers = AnswerCollection()

            ## First Phase: collect and deduplicate all answers across questions in `collected_answers`
            # print("1st")

            for paragraph in query_paragraphs.paragraphs:
                para_id = paragraph.paragraph_id

                for grade in paragraph.retrieve_exam_grade_all(grade_filter=grade_filter):
                    for question_id, answer in grade.answers:
                        question = questions.get(question_id, None)
                        if question is None:
                            raise RuntimeError(f'Query_id {query_id}: Cant obtain question for Question_id {question_id}. Valid question ids are: {questions.keys()}')

                        collected_answers.add_answer(question_id=question_id, answer=answer, para=paragraph)

            ## Second Phase: collect verification data 
            # print("2nd")

            for (question_id, question)  in questions.items():                    
                answers = collected_answers.answers_to_check(question_id=question_id)
                for answer in answers:
                    verification_data = AnswerVerificationData(question=question, answer=answer, llm_response=None, is_correct=None)
                    collected_answers.set_verification_data(question_id=question_id, answer=answer, verification_data=verification_data)
                    # is_correct = verify_answer_with_t5(question, answer)
                    # set_answer_correctness(question_id=question_id, answer=answer, is_correct=is_correct)

            ## Third Phase run LLM
            # print("3rd")
            print("verifying answers...")
            if sample is not None:  # subsample for debugging
                shuffled_verification_data = list(collected_answers.all_verification_data())
                shuffle(shuffled_verification_data)
                t5.verify_answers_with_t5( shuffled_verification_data[0:sample])
            else:
                t5.verify_answers_with_t5( collected_answers.all_verification_data())

            ## Fourth Phase: write checked answers back!
            print("... done")
            # print("4th")
                        
            for paragraph in query_paragraphs.paragraphs:
                para_id = paragraph.paragraph_id

                for grade in paragraph.retrieve_exam_grade_all(grade_filter=grade_filter):
                    correct_answered = list()
                    wrong_answered = list()

                    for question_id, answer in grade.answers:
                        question = questions.get(question_id, None)
                        if question is None:
                            raise RuntimeError(f'Query_id {query_id}: Cant obtain question for Question_id {question_id}. Valid question ids are: {questions.keys()}')

                        qp:QuestionPrompt = tqa_loader.question_obj_to_prompt( q=question, prompt_class=new_prompt_class)

                        is_correct = collected_answers.get_verified_answer(question_id=question_id, answer=answer, para=paragraph)

                        if is_correct is None:
                            if sample is None: # do not print when sampling
                                print(f"Missing data for question_id={question_id}, answer={answer}, para={paragraph.paragraph_id}")
                            pass
                        else:
                            if is_correct:
                                correct_answered.append(question_id)
                            else:
                                wrong_answered.append(question_id)

                    if len(correct_answered)>0:
                        print(f"Query: {query_id} Para: {para_id}:  Correct: {correct_answered}; numWrong: {len(wrong_answered)}")
                    grade.llm_options["answer_match"]=qp.answer_match_info()

                    if grade.prompt_info is not None:
                        grade.prompt_info =  qp.prompt_info(old_prompt_info= (grade.prompt_info))
                    else:
                        grade.prompt_info = qp.prompt_info(old_prompt_info= questionPromptWithChoices_prompt_info)

                    grade.correctAnswered = correct_answered
                    grade.wrongAnswered = wrong_answered
                    grade.exam_ratio = frac(len(correct_answered), len(correct_answered)+len(wrong_answered))
                    pass

            file.write(parse.dumpQueryWithFullParagraphList(query_paragraphs))
            file.flush()
            pass

        # parse.writeQueryWithFullParagraphs(file_path=fixed_graded_query_paragraphs_file
                                        #    , queryWithFullParagraphList=graded_query_paragraphs)
        print(f"fixed written to {fixed_graded_query_paragraphs_file}")
        file.close()


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
    parser.add_argument('--question-set', type=str, choices=["tqa","genq","question-bank"], metavar="SET ", help='Which question set to use. Options: tqa or naghmeh ')
    # parser.add_argument('--testset', type=str, choices=["cary3","dl19"], required=True, metavar="SET ", help='Which question set to use. Options: tqa or naghmeh ')
    parser.add_argument('--answer-verification-prompt', type=str, metavar="PROMPT-CLASS", help='Prompt class to use for answer re-verification')
    parser.add_argument( '--t5-verification', action='store_true', help='If set, will use T5 for answer verification')
    parser.add_argument( '--sample', action='store_true', help='If set, will subsample responses for verification (for debugging)')

    args = parser.parse_args(args=cmdargs)    

    if args.question_set != "tqa":
        raise RuntimeError("Only tqa questions can be re-verified (correct answers are required)")
        
    grade_filter = parse.GradeFilter(model_name=args.model, prompt_class = args.prompt_class, is_self_rated=None, min_self_rating=None, question_set=args.question_set, prompt_type=get_prompt_type_from_prompt_class(args.prompt_class), data_set=None)


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


    if(args.t5_verification):
        answer_verification_prompt = "QuestionCompleteConcisePromptWithT5VerifiedAnswerKey2"
        if not args.answer_verification_prompt is None:
            answer_verification_prompt = args.answer_verification_prompt

        t5_verify_answers(args.out, graded_query_paragraphs, grade_filter, answer_verification_prompt, query2questions, sample = 100 if args.sample else None)
    else:
        answer_verification_prompt = "QuestionCompleteConcisePromptWithAnswerKey2"
        if not args.answer_verification_prompt is None:
            answer_verification_prompt = args.answer_verification_prompt

        reverify_answers(args.out, graded_query_paragraphs, grade_filter, args.answer_verification_prompt, query2questions)
    pass



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
    grade_filter = parse.GradeFilter(model_name="google/flan-t5-large", prompt_class=None, is_self_rated=None, min_self_rating=None, question_set="tqa", prompt_type=get_prompt_type_from_prompt_class(args.prompt_class), data_set=None)

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


    