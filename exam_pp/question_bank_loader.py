from abc import abstractmethod
import abc
from collections import defaultdict
import hashlib
import itertools
import time
import typing
from pydantic import BaseModel
from pydantic.generics import GenericModel
import json
from typing import List, Any, Optional, Dict, Set, TextIO, Tuple,  TypeVar, Generic, List, cast
from dataclasses import dataclass
import gzip
from pathlib import Path

from .query_loader import direct_grading_prompt

from .test_bank_prompts import DirectGradingPrompt, FagB, NuggetExtractionPrompt, NuggetSelfRatedPrompt, Prompt, QuestionAnswerablePromptWithChoices, QuestionCompleteConciseUnanswerablePromptWithChoices, QuestionPrompt, QuestionSelfRatedUnanswerablePromptWithChoices
from .pydantic_helper import pydantic_dump


class TestPoint(BaseModel):
    query_id :str
    facet_id: Optional[str]
    info: Optional[Any]


T = TypeVar('T', bound=TestPoint)

class ExamQuestion(TestPoint):
    question_id: str
    question_text: str
    gold_answers: Optional[Set[str]]

class Nugget(TestPoint):
    nugget_id: str
    nugget_text: str

class QueryTestBank(GenericModel, Generic[T]):
    query_id: str
    facet_id: Optional[str]
    facet_text: Optional[str]
    test_collection: str
    query_text: str
    info: Optional[Any]
    items: List[T]



class QueryQuestionBank(QueryTestBank[ExamQuestion]):
    hash:int = 1243234 # TODO random

    def get_questions(self) -> List[ExamQuestion]:
        return self.items

class QueryNuggetBank(QueryTestBank[Nugget]):
    hash:int = 1243234 # TODO random

    def get_nuggets(self) -> List[Nugget]:
        return self.items




def parseTestBank(file_path:Path, use_nuggets:bool) -> typing.Sequence[QueryTestBank] :
    '''Load QueryTestBank (exam questions or nuggets)'''
    # Open the gzipped file

    if(use_nuggets):
        result_n:List[QueryNuggetBank] = list()
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            result_n = [QueryNuggetBank.parse_raw(line) for line in file]
        return result_n

    else: # exam questions

        result_q:List[QueryQuestionBank] = list()
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            result_q = [QueryQuestionBank.parse_raw(line) for line in file]
        return result_q



def writeTestBank(file_path:Path, queryTestBanks:List[QueryTestBank]) :
    # Open the gzipped file
    with gzip.open(file_path, 'wt', encoding='utf-8') as file:
        # Iterate over each line in the file
        for bank in queryTestBanks:
            file.write(pydantic_dump(bank)+'\n')

## --------- create question bank ------------------

def write_single_query_test_bank(file, bank:QueryTestBank):
    file.write(pydantic_dump(bank)+"\n")
    file.flush()


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



def as_exam_test_points(query_id:str, facet_id:Optional[str], test_texts:List[str], generation_info:Any, use_nuggets:bool)->List[TestPoint]:
    result:List[TestPoint] = list()
    use_facet_id = facet_id is not None
    for test_text in test_texts:
        clean_test_text = test_text.strip()
        test_hash = get_md5_hash(clean_test_text)

        test_id = f'{query_id}/{facet_id}/{test_hash}' if use_facet_id else f'{query_id}/{test_hash}'

        if use_nuggets:
            n = Nugget(nugget_id=test_id
                            , query_id=query_id
                            , nugget_text=clean_test_text
                            , facet_id=facet_id if use_facet_id else None
                            , info=generation_info)
            result.append(n)
        else: # exam question 
            q = ExamQuestion(question_id=test_id
                            , query_id=query_id
                            , question_text=clean_test_text
                            , facet_id=facet_id if use_facet_id else None
                            , info=generation_info
                            , gold_answers= None)
            result.append(q)
    return result


def emit_test_bank_entry(out_file:TextIO, test_collection:str, generation_info:Any, query_id:str, query_facet_id:Optional[str], query_facet_text:Optional[str], query_text:str, question_texts:List[str], use_nuggets:bool):
    test_points:List[TestPoint]
    test_points = as_exam_test_points(query_id=query_id
                                               , facet_id=query_facet_id
                                               , test_texts=question_texts
                                            #    , generation_info=generation_info
                                               , generation_info = None
                                               , use_nuggets=use_nuggets)

    if not use_nuggets:

        question_bank = QueryQuestionBank(items=[cast(ExamQuestion, q) for q in test_points]
                                        , query_id=query_id
                                        , facet_id=query_facet_id
                                        , facet_text=query_facet_text
                                        , test_collection=test_collection
                                        , query_text=query_text
                                        , info=generation_info
                                        )
        write_single_query_test_bank(file=out_file, bank=question_bank)


    elif use_nuggets:
        nugget_bank = QueryNuggetBank(items=[cast(Nugget, n) for n in test_points]
                                        , query_id=query_id
                                        , facet_id=query_facet_id
                                        , facet_text=query_facet_text
                                        , test_collection=test_collection
                                        , query_text=query_text
                                        , info=generation_info
                                        )
        write_single_query_test_bank(file=out_file, bank=nugget_bank)


## -------------------------------------------
        

def load_prompts_from_test_bank(question_file:Path, use_nuggets:bool, prompt_class:str="QuestionPromptWithChoices")-> List[Tuple[str, List[Prompt]]]:
    '''Iterate over all test bank entries, first try to load as direct grading prompt, if that fails check for the `use_nuggets` flag and try to load as question or nugget prompts.'''
    def generate_letter_choices() -> Set[str]:
        char_options = ['A','B','C','D', 'a', 'b', 'c', 'd', 'i', 'ii', 'iii', 'iv']
        option_non_answers = set( itertools.chain.from_iterable([[f'{ch})', f'({ch})', f'[{ch}]', f'{ch}.',f'{ch}']   for ch in char_options]) )
        return option_non_answers
        
    option_non_answers = generate_letter_choices()
    option_non_answers.add('unanswerable')
    test_banks = parseTestBank(question_file, use_nuggets=use_nuggets)
    prompt_dict : Dict[str, List[Prompt]]
    prompt_dict = defaultdict(list)
    prompt:Prompt

    for bank in test_banks:

        try:
            # prompt = direct_grading_prompt(prompt_class=prompt_class, query_id=bank.query_id, query_text=bank.query_text, facet_id=bank.facet_id, facet_text=bank.facet_text)
            prompt = direct_grading_prompt(prompt_class=prompt_class, query_id=bank.query_id, query_text=bank.query_text, facet_id=None, facet_text=None)
            # hack to only include one direct prompt for each query.

            if bank.query_id not in prompt_dict:  # ONLY one direct grading prompt per query!!! 
                prompt_dict[bank.query_id].append(prompt)
            else:
                pass # if we had direct grading prompts that would use facets those could be added here.

        except:
            # not a direct grading prompt
            # try question and nugget prompts


            if not use_nuggets:
                question_bank = cast(QueryQuestionBank, bank)
                query_id = question_bank.query_id
                for question in question_bank.get_questions():
                    if not question.query_id == query_id:
                            raise RuntimeError(f"query_ids don't match between QueryQuestionBank ({query_id}) and contained ExamQuestion ({question.query_id}) ")
                    if(prompt_class =="QuestionSelfRatedUnanswerablePromptWithChoices"):
                        prompt = QuestionSelfRatedUnanswerablePromptWithChoices(question_id = question.question_id
                                                                            , question = question.question_text
                                                                            , query_id = question_bank.query_id
                                                                            , facet_id = question.facet_id
                                                                            , query_text = question_bank.query_text
                                                                            , unanswerable_expressions = option_non_answers
                                                                            )
                    elif(prompt_class == "QuestionCompleteConciseUnanswerablePromptWithChoices"):
                        prompt = QuestionCompleteConciseUnanswerablePromptWithChoices(question_id = question.question_id
                                                                            , question = question.question_text
                                                                            , query_id = question_bank.query_id
                                                                            , facet_id = question.facet_id
                                                                            , query_text = question_bank.query_text
                                                                            , unanswerable_expressions = option_non_answers
                                                                            )
                    elif(prompt_class == "QuestionAnswerablePromptWithChoices"):
                        prompt = QuestionAnswerablePromptWithChoices(question_id = question.question_id
                                                                            , question = question.question_text
                                                                            , query_id = question_bank.query_id
                                                                            , facet_id = question.facet_id
                                                                            , query_text = question_bank.query_text
                                                                            , unanswerable_expressions = option_non_answers
                                                                            )
                    else:
                        raise RuntimeError(f"Prompt class {prompt_class} not supported by this question_loader or direct_grading loader.")\
                

                    prompt_dict[query_id].append(prompt)


            if use_nuggets:
                nugget_bank = cast(QueryNuggetBank, bank)
                query_id = nugget_bank.query_id
                for nugget in nugget_bank.get_nuggets():
                    if not nugget.query_id == query_id:
                            raise RuntimeError(f"query_ids don't match between QueryNuggetBank ({query_id}) and contained ExamNugget ({nugget.query_id}) ")
                    
                    if(prompt_class =="NuggetSelfRatedPrompt"):
                        prompt = NuggetSelfRatedPrompt(nugget_id = nugget.nugget_id
                                                    , nugget_text = nugget.nugget_text
                                                    , query_id = nugget.query_id
                                                    , facet_id = nugget.facet_id
                                                    , query_text = nugget_bank.query_text
                                                    , unanswerable_expressions = option_non_answers
                                                    )
                    elif(prompt_class =="NuggetExtractionPrompt"):
                        prompt = NuggetExtractionPrompt(nugget_id = nugget.nugget_id
                                                    , nugget_text = nugget.nugget_text
                                                    , query_id = nugget.query_id
                                                    , facet_id = nugget.facet_id
                                                    , query_text = nugget_bank.query_text
                                                    , unanswerable_expressions = option_non_answers
                                                    )
                    else:
                        raise RuntimeError(f"Prompt class {prompt_class} not supported by this nugget_loader  or direct_grading loader.")\
                

                    prompt_dict[query_id].append(prompt)



    return list(prompt_dict.items())


## -------------------------------------------        

def main():
    question1 = ExamQuestion(question_id="12897q981", query_id="Q1", question_text="What was my question, again?", facet_id=None, info=None, gold_answers=None)
    question2 = ExamQuestion(question_id="42", query_id="Q1", question_text="Who am I?", facet_id="some_facet", info=None, gold_answers=None)

    print(pydantic_dump(question1))
    queryBank = QueryQuestionBank(query_id="Q1", facet_id=None, test_collection="dummy", query_text="everything", info=None
                      , items = [question1, question2]
                      )


    writeTestBank("newfile.jsonl.gz", [queryBank, queryBank])

    bank_again = parseTestBank("newfile.jsonl.gz")
    print(bank_again[0])

    prompts = load_prompts_from_test_bank("newfile.jsonl.gz", prompt_class="QuestionSelfRatedUnanswerablePromptWithChoices")
    print(prompts[0])

if __name__ == "__main__":
    main()

