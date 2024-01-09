import itertools
import os
from pathlib import Path
from typing import Tuple, List, Any, Dict, Optional
import json


from question_types import QuestionPromptWithChoices, QuestionSelfRatedUnanswerablePromptWithChoices, QuestionCompleteConcisePromptWithAnswerKey ,QuestionPrompt
from question_types import *


def loadTQA(tqa_file:Path, prompt_class:str="QuestionPromptWithChoices")-> List[Tuple[str, List[QuestionPrompt]]]:

    result:List[Tuple[str,List[QuestionPrompt]]] = list()

    file = open(tqa_file)
    for lesson in json.load(file):
        local_results:List[QuestionPrompt] = list()
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
           
            qpc:QuestionPrompt
            if(prompt_class =="QuestionSelfRatedUnanswerablePromptWithChoices"):
                qpc = QuestionSelfRatedUnanswerablePromptWithChoices(question_id=qid, question=question, query_id=query_id, facet_id = None, query_text=query_text, unanswerable_expressions=set())
            elif(prompt_class == "QuestionCompleteConcisePromptWithAnswerKey"):
                qpc = QuestionCompleteConcisePromptWithAnswerKey(question_id=qid, question=question,choices=choices, correct=correct, correctKey = correctKey, query_id=query_id, facet_id = None, query_text=query_text)
            elif(prompt_class == "QuestionPromptWithChoices"):
                qpc = QuestionPromptWithChoices(question_id=qid, question=question,choices=choices, correct=correct, correctKey = correctKey, query_id=query_id, facet_id = None, query_text=query_text)
            else:
                raise RuntimeError(f"Prompt class {prompt_class} not supported by tqa_loader.")
            # print('qpc', qpc)
            local_results.append(qpc)
        result.append((query_id, local_results))

    return result
            

def load_all_tqa_data(tqa_path:Path = Path("./tqa_train_val_test"), prompt_class:str="QuestionPromptWithChoices"):
    return list(itertools.chain(
            loadTQA(tqa_path.joinpath('train','tqa_v1_train.json'), prompt_class=prompt_class)
            , loadTQA(tqa_path.joinpath('val','tqa_v1_val.json'), prompt_class=prompt_class)     
            , loadTQA(tqa_path.joinpath('test','tqa_v2_test.json'), prompt_class=prompt_class) 
            ))
    

def main():
    """Emit one question"""
    print("hello")
    questions = load_all_tqa_data()
    print("num questions loaded: ", len(questions))
    print("q0",questions[0])
    
    # qa = McqaPipeline()
    # answerQuestions(questions, qa)


if __name__ == "__main__":
    main()
