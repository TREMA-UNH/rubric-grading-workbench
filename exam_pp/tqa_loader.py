import itertools
import os
from pathlib import Path
from typing import Tuple, List, Any, Dict, Optional

from pydantic.v1 import BaseModel
import json

from .question_bank_loader import ExamQuestion, QueryQuestionBank

from .query_loader import direct_grading_prompt

from .pydantic_helper import pydantic_dump
from .test_bank_prompts import QuestionPromptWithChoices, QuestionSelfRatedUnanswerablePromptWithChoices, QuestionCompleteConcisePromptWithAnswerKey ,QuestionPrompt
from .test_bank_prompts import *


def fix_tqa_car_query_id(tqa_query_id:str)->str:
    return f'tqa2:{tqa_query_id}'

# @dataclass
class Question(BaseModel):
    query_id:str
    query_text:Optional[str]
    qid:str
    question:str
    choices:Optional[Dict[str,str]]
    correctKey:Optional[str]
    correct:Optional[str]
    facet_id:Optional[str]



def writeQuestions(file_path:Path, questions:List[Tuple[str, List[Question]]]) :
    import gzip
    # Open the gzipped file
    with gzip.open(file_path, 'wt', encoding='utf-8') as file:
        # Iterate over each line in the file
        for queryId, qs in questions:
                for q in qs:
                    j = pydantic_dump(q)
                    file.write(j)
                    file.write('\n')
        # file.writelines([json.dumps(x)+'\n' for x in questions])
    file.close()


def parseConvertedQuestions(file_path:Path)->List[Question]:
    import gzip

    result:List[Question] = list()
    try: 
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            # return [parseQueryWithFullParagraphList(line) for line in file]
            for line in file:
                result.append(Question.parse_raw(line))
    except  EOFError as e:
        print("Warning: Gzip EOFError on {file_path}. Use truncated data....\nFull Error:\n{e}")
    return result






#  ------------------------

def load_TQA_questions(tqa_file:Path)-> List[Tuple[str, List[Question]]]:

    result:List[Tuple[str,List[Question]]] = list()

    file = open(tqa_file)
    for lesson in json.load(file):
        local_results:List[Question] = list()
        query_id = lesson['globalID']
        query_text = lesson['lessonName']

        for qid, q in lesson['questions']['nonDiagramQuestions'].items():
            question:str = q['beingAsked']['processedText']
            choices:Dict[str,str] = {key: x['processedText'] for key,x in q['answerChoices'].items() }
            correctKey:str = q['correctAnswer']['processedText']
            correct:Optional[str] = choices.get(correctKey)

            if correct is None:
                print('bad question, because correct answer is not among the choices', 'key: ',correctKey, 'choices: ', choices)
                continue
           
            q = Question(query_id=query_id
                         , query_text=query_text
                         , qid=qid
                         , question=question
                         , choices=choices
                         , correctKey=correctKey
                         , correct=correct
                         , facet_id=None)
            # print('qpc', qpc)
            local_results.append(q)
        result.append((query_id, local_results))

    return result


def question_obj_to_prompt(prompt_class:str, q:Question, self_rater_tolerant:bool)->QuestionPrompt:
    return question_to_prompt(prompt_class
                            , query_id=q.query_id
                            , query_text=q.query_text
                            , qid=q.qid
                            , question=q.question
                            , choices=q.choices
                            , correctKey=q.correctKey
                            , correct={q.correct} if q.correct is not None else set()
                            , facet_id = q.facet_id
                            , self_rater_tolerant= self_rater_tolerant
                            )

def question_to_prompt(prompt_class, query_id, query_text, qid, question, choices:Optional[Dict[str,str]], correctKey, correct:Set[str], facet_id, self_rater_tolerant:bool)->QuestionPrompt:
    qpc:QuestionPrompt

    if(prompt_class =="QuestionSelfRatedUnanswerablePromptWithChoices"):
        qpc = QuestionSelfRatedUnanswerablePromptWithChoices(question_id=qid
                                                                     , question=question
                                                                     , query_id=query_id
                                                                     , facet_id = facet_id
                                                                     , query_text=query_text
                                                                     , unanswerable_expressions=set()
                                                                     , self_rater_tolerant=self_rater_tolerant
                                                                     )
    elif(prompt_class == "QuestionCompleteConcisePromptWithAnswerKey"):
        if choices is None:
            raise RuntimeError(f"{prompt_class}: Choices must not be null")
        qpc = QuestionCompleteConcisePromptWithAnswerKey(question_id=qid
                                                                 , question=question
                                                                 , choices=choices
                                                                 , correct=correct
                                                                 , correctKey = correctKey
                                                                 , query_id=query_id
                                                                 , facet_id = facet_id
                                                                 , query_text=query_text
                                                                 )
    elif(prompt_class == "QuestionCompleteConcisePromptWithAnswerKey2"):
        if choices is None:
            raise RuntimeError(f"{prompt_class}: Choices must not be null")
        qpc = QuestionCompleteConcisePromptWithAnswerKey2(question_id=qid
                                                                 , question=question
                                                                 , choices=choices
                                                                 , correct=correct
                                                                 , correctKey = correctKey
                                                                 , query_id=query_id
                                                                 , facet_id = facet_id
                                                                 , query_text=query_text
                                                                 ) 
    elif(prompt_class == "QuestionCompleteConciseUnanswerablePromptWithChoices"):
        qpc = QuestionCompleteConciseUnanswerablePromptWithChoices(question_id=qid
                                                                           , question=question
                                                                           , query_id=query_id
                                                                           , facet_id = facet_id
                                                                           , query_text=query_text
                                                                           , unanswerable_expressions=set()
                                                                           )
    elif(prompt_class == "QuestionPromptWithChoices"):
        if choices is None:
            raise RuntimeError(f"{prompt_class}: Choices must not be null")
        qpc = QuestionPromptWithChoices(question_id=qid
                                                , question=question
                                                , choices=choices
                                                , correct=correct
                                                , correctKey = correctKey
                                                , query_id=query_id
                                                , facet_id = facet_id
                                                , query_text=query_text
                                                )
    elif(prompt_class == "QuestionCompleteConcisePromptWithT5VerifiedAnswerKey2"):
        if choices is None:
            raise RuntimeError(f"{prompt_class}: Choices must not be null")
        qpc = QuestionCompleteConcisePromptWithT5VerifiedAnswerKey2(question_id=qid
                                                                 , question=question
                                                                 , choices=choices
                                                                 , correct=correct
                                                                 , correctKey = correctKey
                                                                 , query_id=query_id
                                                                 , facet_id = facet_id
                                                                 , query_text=query_text
                                                                 )     
    else:

        raise RuntimeError(f"Prompt class {prompt_class} not supported by tqa_loader.")
    return qpc
            

def load_TQA_prompts(self_rater_tolerant:bool,tqa_file:Path, prompt_class:str="QuestionPromptWithChoices")-> List[Tuple[str, List[Prompt]]]:
    query_questions = load_TQA_questions(tqa_file)

    if get_prompt_type_from_prompt_class(prompt_class) == DirectGradingPrompt.my_prompt_type:
        # Direct grading prompt

        # hack to only include one direct prompt for each query.
        # prompt = direct_grading_prompt(prompt_class=prompt_class, query_id=bank.query_id, query_text=bank.query_text, facet_id=bank.facet_id, facet_text=bank.facet_text)
        return [(query_id,  [
                             direct_grading_prompt(prompt_class=prompt_class, query_id=query_id, query_text=questions[0].query_text, facet_id=None, facet_text=None, self_rater_tolerant=self_rater_tolerant)
                            ])  
                for query_id, questions in query_questions if len(questions)>0]


    if get_prompt_type_from_prompt_class(prompt_class) == QuestionPrompt.my_prompt_type:
        # Question Prompt
        return [(query,  [question_obj_to_prompt(prompt_class, q=question, self_rater_tolerant=self_rater_tolerant)  
                         for question in questions])  
                                for query, questions in query_questions]

    else:
        raise RuntimeError(f"prompt {prompt_class} not supported by TQA")


def load_all_tqa_data(self_rater_tolerant:bool, tqa_path:Path = Path("./tqa_train_val_test"), prompt_class:str="QuestionPromptWithChoices"):
    return list(itertools.chain(
            load_TQA_prompts(tqa_file=tqa_path.joinpath('train','tqa_v1_train.json'), prompt_class=prompt_class, self_rater_tolerant=self_rater_tolerant)
            , load_TQA_prompts(tqa_file=tqa_path.joinpath('val','tqa_v1_val.json'), prompt_class=prompt_class, self_rater_tolerant=self_rater_tolerant)     
            , load_TQA_prompts(tqa_file=tqa_path.joinpath('test','tqa_v2_test.json'), prompt_class=prompt_class, self_rater_tolerant=self_rater_tolerant) 
            ))


def load_all_tqa_questions(tqa_path:Path = Path("./tqa_train_val_test")):
    return list(itertools.chain(
            load_TQA_questions(tqa_path.joinpath('train','tqa_v1_train.json'))
            , load_TQA_questions(tqa_path.joinpath('val','tqa_v1_val.json'))     
            , load_TQA_questions(tqa_path.joinpath('test','tqa_v2_test.json')) 
            ))
    
#  -----


def parseTestBank_all(tqa_path:Path = Path("./tqa_train_val_test"), fix_query_id:Optional[Callable[[str], str]]=None):
    return list(itertools.chain(
            parseTestBank(tqa_path.joinpath('train','tqa_v1_train.json'),fix_query_id=fix_query_id)
            , parseTestBank(tqa_path.joinpath('val','tqa_v1_val.json'),fix_query_id=fix_query_id)     
            , parseTestBank(tqa_path.joinpath('test','tqa_v2_test.json'),fix_query_id=fix_query_id) 
            ))

def parseTestBank(file_path:Path, fix_query_id:Optional[Callable[[str], str]]=None) -> Sequence[QueryQuestionBank] :
    '''Load as QueryTestBank (exam questions or nuggets)'''

    result: List[QueryQuestionBank] = list()

    for (query_id, questions) in load_TQA_questions(file_path):
        fixed_query_id = query_id if fix_query_id is None else fix_query_id(query_id)

        # -> List[Tuple[str, List[Question]]]:
        if len(questions):
            exam_questions = list()
        
            for question in questions:
                gold_answer_set = set(question.correct) if question.correct is not None else set()
                exam_question = ExamQuestion(question_id=question.qid, question_text=question.question, gold_answers=gold_answer_set, facet_id=None, info=None, query_id=question.query_id)
                exam_questions.append(exam_question)

            query_text = questions[0].query_text if questions[0].query_text is not None else ""
            
            question_bank = QueryQuestionBank(query_id= fixed_query_id, facet_id= None, facet_text=None, test_collection="tqa", query_text=query_text, info=None, items=exam_questions, hash=9999)
            result.append(question_bank)

    return result


#  -----

def main():
    """Emit one question"""
    print("hello")
    questions = load_all_tqa_data(self_rater_tolerant=False)
    print("num questions loaded: ", len(questions))
    print("q0",questions[0])


if __name__ == "__main__":
    main()
