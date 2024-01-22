import itertools
import os
from pathlib import Path
from typing import Tuple, List, Any, Dict, Optional
import json


from .question_types import QuestionPromptWithChoices, QuestionSelfRatedUnanswerablePromptWithChoices, QuestionCompleteConcisePromptWithAnswerKey ,QuestionPrompt
from .question_types import *



@dataclass
class Question():
    query_id:str
    query_text:Optional[str]
    qid:str
    question:str
    choices:Optional[Dict[str,str]]
    correctKey:Optional[str]
    correct:Optional[str]


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
           
            q = Question(query_id, query_text, qid, question, choices, correctKey, correct)
            # print('qpc', qpc)
            local_results.append(q)
        result.append((query_id, local_results))

    return result


def question_obj_to_prompt(prompt_class:str, q:Question)->QuestionPrompt:
    return question_to_prompt(prompt_class
                            , query_id=q.query_id
                            , query_text=q.query_text
                            , qid=q.qid
                            , question=q.question
                            , choices=q.choices
                            , correctKey=q.correctKey
                            , correct=q.correct
                            )

def question_to_prompt(prompt_class, query_id, query_text, qid, question, choices:Optional[Dict[str,str]], correctKey, correct)->QuestionPrompt:
    qpc:QuestionPrompt

    if(prompt_class =="QuestionSelfRatedUnanswerablePromptWithChoices"):
        qpc = QuestionSelfRatedUnanswerablePromptWithChoices(question_id=qid
                                                                     , question=question
                                                                     , query_id=query_id
                                                                     , facet_id = None
                                                                     , query_text=query_text
                                                                     , unanswerable_expressions=set()
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
                                                                 , facet_id = None
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
                                                                 , facet_id = None
                                                                 , query_text=query_text
                                                                 ) 
    elif(prompt_class == "QuestionCompleteConciseUnanswerablePromptWithChoices"):
        qpc = QuestionCompleteConciseUnanswerablePromptWithChoices(question_id=qid
                                                                           , question=question
                                                                           , query_id=query_id
                                                                           , facet_id = None
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
                                                , facet_id = None
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
                                                                 , facet_id = None
                                                                 , query_text=query_text
                                                                 )     
    else:
        raise RuntimeError(f"Prompt class {prompt_class} not supported by tqa_loader.")
    return qpc
            

def load_TQA_prompts(tqa_file:Path, prompt_class:str="QuestionPromptWithChoices")-> List[Tuple[str, List[QuestionPrompt]]]:
    query_questions = load_TQA_questions(tqa_file)
    return [(query,  [question_obj_to_prompt(prompt_class, q=question)  
                         for question in questions])  
                                for query, questions in query_questions]


def load_all_tqa_data(tqa_path:Path = Path("./tqa_train_val_test"), prompt_class:str="QuestionPromptWithChoices"):
    return list(itertools.chain(
            load_TQA_prompts(tqa_path.joinpath('train','tqa_v1_train.json'), prompt_class=prompt_class)
            , load_TQA_prompts(tqa_path.joinpath('val','tqa_v1_val.json'), prompt_class=prompt_class)     
            , load_TQA_prompts(tqa_path.joinpath('test','tqa_v2_test.json'), prompt_class=prompt_class) 
            ))


def load_all_tqa_questions(tqa_path:Path = Path("./tqa_train_val_test")):
    return list(itertools.chain(
            load_TQA_questions(tqa_path.joinpath('train','tqa_v1_train.json'))
            , load_TQA_questions(tqa_path.joinpath('val','tqa_v1_val.json'))     
            , load_TQA_questions(tqa_path.joinpath('test','tqa_v2_test.json')) 
            ))
    


def main():
    """Emit one question"""
    print("hello")
    questions = load_all_tqa_data()
    print("num questions loaded: ", len(questions))
    print("q0",questions[0])


if __name__ == "__main__":
    main()
