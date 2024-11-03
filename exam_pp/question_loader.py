import itertools
from typing import Tuple, List, Any, Dict, Optional, Set
from pydantic.v1 import BaseModel
import json
import re
import json
import itertools
from pathlib import Path
import hashlib


from .tqa_loader import Question, writeQuestions, parseConvertedQuestions


from .test_bank_prompts import QuestionAnswerablePromptWithChoices, QuestionPrompt, QuestionSelfRatedUnanswerablePromptWithChoices, QuestionCompleteConciseUnanswerablePromptWithChoices
from .test_bank_prompts import *


def get_md5_hash(input_string: str) -> str:
    # Convert the string to bytes
    input_bytes = input_string.encode('utf-8')

    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Update the hash object with the bytes
    md5_hash.update(input_bytes)

    # Get the hexadecimal digest of the hash
    hex_digest = md5_hash.hexdigest()

    return hex_digest

def strip_enumeration(s: str) -> str:
    # Regex pattern to match enumeration at the start of the string
    # This pattern matches one or more digits followed by a dot and a space
    pattern = r'^\d+\.\s+'
    # Substitute the matched pattern with an empty string
    return re.sub(pattern, '', s)

# def int_key_to_str(i:int)->str:
#     return f"chr(65+i)"

def generate_letter_choices() -> Set[str]:
    char_options = ['A','B','C','D', 'a', 'b', 'c', 'd', 'i', 'ii', 'iii', 'iv']
    option_non_answers = set( itertools.chain.from_iterable([[f'{ch})', f'({ch})', f'[{ch}]', f'{ch}.',f'{ch}']   for ch in char_options]) )
    return option_non_answers
    

def load_naghmehs_questions(question_file:Path)-> List[Tuple[str, List[Question]]]:

    result:List[Tuple[str,List[Question]]] = list()

    # option_non_answers = generate_letter_choices()
    # option_non_answers.add('unanswerable')

    content = json.load(open(question_file))
    for query_id, data in content.items():
        question_list = list()
        for facet_id, facet_data in data.items():
            query_text = f'{facet_data["title"]} / {facet_data["facet"]}'
            for question_text_orig in facet_data["questions"]:
                question_text = strip_enumeration(question_text_orig)
                if len(question_text)>1:
                    question_hash = get_md5_hash(question_text_orig)
                    qid = f'{query_id}/{facet_id}/{question_hash}'

                    q:Question
                    q = Question(qid = qid
                                , question = question_text
                                , query_id = query_id
                                , facet_id = facet_id
                                , query_text = query_text
                                , choices = None, correctKey=None, correct=None
                                )
                    question_list.append(q)
        result.append((query_id, question_list))
    return result


def load_naghmehs_question_prompts(question_file:Path, prompt_class:str="QuestionPromptWithChoices")-> List[Tuple[str, List[QuestionPrompt]]]:
    # we could also use tqa_loader.question_to_prompt
    question_bank = load_naghmehs_questions(question_file=question_file)

    result:List[Tuple[str,List[QuestionPrompt]]] = list()

    option_non_answers = generate_letter_choices()
    option_non_answers.add('unanswerable')

    q:Question
    for query_id, questions in question_bank:
        qpc_list = list()
        for q in questions:
                    qid = q.qid
                    question_text = q.question
                    query_id = q.query_id
                    facet_id = q.facet_id
                    query_text = q.query_text
                    if(query_text is None):
                        raise RuntimeError("Cannot build prompt from question without question text:"+q)
                    unanswerable_expressions = option_non_answers
                    qpc:QuestionPrompt
                    if(prompt_class =="QuestionSelfRatedUnanswerablePromptWithChoices"):
                        qpc = QuestionSelfRatedUnanswerablePromptWithChoices(question_id = qid
                                                                             , question = question_text
                                                                             , query_id = query_id
                                                                             , facet_id = facet_id
                                                                             , query_text = query_text
                                                                             , unanswerable_expressions = unanswerable_expressions
                                                                             )
                    elif(prompt_class == "QuestionCompleteConciseUnanswerablePromptWithChoices"):
                        qpc = QuestionCompleteConciseUnanswerablePromptWithChoices(question_id=qid
                                                    , question = question_text
                                                    , query_id = query_id
                                                    , facet_id = facet_id
                                                    , query_text = query_text
                                                    , unanswerable_expressions=unanswerable_expressions
                                                    )
                    elif(prompt_class == "QuestionAnswerablePromptWithChoices"):
                        qpc = QuestionAnswerablePromptWithChoices(question_id = qid
                                                    , question= question_text
                                                    , query_id= query_id
                                                    , facet_id = facet_id
                                                    , query_text = query_text
                                                    , unanswerable_expressions=unanswerable_expressions
                                                    )
                    else:
                        raise RuntimeError(f"Prompt class {prompt_class} not supported by naghmeh's question_loader.")\
                

                    qpc_list.append(qpc)
        result.append((query_id, qpc_list))
    return result


def main():
    naghmehs_questions = "./naghmeh-questions.json"
    outfile = "./naghmeh-questions-converted.jsonl.gz"
    qs = load_naghmehs_questions(question_file=naghmehs_questions)
    writeQuestions(outfile, qs)

    # xs = parseConvertedQuestions(outfile)
    # print(xs[0])


if __name__ == "__main__":
    main()
