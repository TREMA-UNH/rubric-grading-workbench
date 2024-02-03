from collections import defaultdict
import hashlib
import itertools
import typing
from pydantic import BaseModel
import json
from typing import List, Any, Optional, Dict, Set, Tuple
from dataclasses import dataclass
import gzip
from pathlib import Path

from .question_types import QuestionAnswerablePromptWithChoices, QuestionCompleteConciseUnanswerablePromptWithChoices, QuestionPrompt, QuestionSelfRatedUnanswerablePromptWithChoices
from .pydantic_helper import pydantic_dump

class ExamQuestion(BaseModel):
    question_id: str
    question_text: str
    query_id :str
    facet_id: Optional[str]
    info: Optional[Any]

class QueryQuestionBank(BaseModel):
    query_id: str
    facet_id: Optional[str]
    test_collection: str
    query_text: str
    info: Optional[Any]
    questions: List[ExamQuestion]


# Path to the benchmarkY3test-qrels-with-text.jsonl.gz file
def parseQuestionBank(file_path:Path) -> typing.List[QueryQuestionBank] :
    '''Load JSONL.GZ file with exam questions'''
    # Open the gzipped file

    result:List[QueryQuestionBank] = list()
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        result = [QueryQuestionBank.parse_raw(line) for line in file]
    return result



def writeQuestionBank(file_path:Path, queryQuestionBanks:List[QueryQuestionBank]) :
    # Open the gzipped file
    with gzip.open(file_path, 'wt', encoding='utf-8') as file:
        # Iterate over each line in the file
        for questionBank in queryQuestionBanks:
            file.write(pydantic_dump(questionBank)+'\n')

## --------- create question bank ------------------

def write_single_query_question_bank(file, questionBank:QueryQuestionBank):
    file.write(pydantic_dump(questionBank)+"\n")
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



def as_exam_questions(query_id:str, facet_id:Optional[str], questions:List[str], generation_info:Any)->List[ExamQuestion]:
    result:List[ExamQuestion] = list()
    use_facet_id = facet_id is not None
    for question in questions:
        clean_question = question.strip()
        question_hash = get_md5_hash(clean_question)

        qid = f'{query_id}/{facet_id}/{question_hash}' if use_facet_id else f'{query_id}/{question_hash}'

        q = ExamQuestion(question_id=qid
                         , query_id=query_id
                         , question_text=clean_question
                         , facet_id=facet_id if use_facet_id else None
                         , info=generation_info)
        result.append(q)
    return result


def emit_exam_question_bank_entry(out_file, test_collection, generation_info, query_id, query_facet_id, query_text, questions):
    exam_questions = as_exam_questions(query_id=query_id
                                               , facet_id=query_facet_id
                                               , questions=questions
                                               , generation_info=generation_info)
            
    question_bank = QueryQuestionBank(questions=exam_questions
                                    , query_id=query_id
                                    , facet_id=query_facet_id
                                    , test_collection=test_collection
                                    , query_text=query_text
                                    , info = None
                                    # , info=generation_info
                                    )
    write_single_query_question_bank(file=out_file, questionBank=question_bank)


## -------------------------------------------
        

def load_exam_question_bank(question_file:Path, prompt_class:str="QuestionPromptWithChoices")-> List[Tuple[str, List[QuestionPrompt]]]:
    def generate_letter_choices() -> Set[str]:
        char_options = ['A','B','C','D', 'a', 'b', 'c', 'd', 'i', 'ii', 'iii', 'iv']
        option_non_answers = set( itertools.chain.from_iterable([[f'{ch})', f'({ch})', f'[{ch}]', f'{ch}.',f'{ch}']   for ch in char_options]) )
        return option_non_answers
        
    option_non_answers = generate_letter_choices()
    option_non_answers.add('unanswerable')
    content = parseQuestionBank(question_file)
    qpc_dict = defaultdict(list)
    for queryQuestions in content:
        query_id = queryQuestions.query_id
        for question in queryQuestions.questions:
            if not question.query_id == query_id:
                    raise RuntimeError(f"query_ids don't match between QueryQuestionBank ({query_id}) and contained ExamQuestion ({question.query_id}) ")
            qpc:QuestionPrompt
            if(prompt_class =="QuestionSelfRatedUnanswerablePromptWithChoices"):
                qpc = QuestionSelfRatedUnanswerablePromptWithChoices(question_id = question.question_id
                                                                    , question = question.question_text
                                                                    , query_id = question.query_id
                                                                    , facet_id = question.facet_id
                                                                    , query_text = queryQuestions.query_text
                                                                    , unanswerable_expressions = option_non_answers
                                                                    )
            elif(prompt_class == "QuestionCompleteConciseUnanswerablePromptWithChoices"):
                qpc = QuestionCompleteConciseUnanswerablePromptWithChoices(question_id = question.question_id
                                                                    , question = question.question_text
                                                                    , query_id = question.query_id
                                                                    , facet_id = question.facet_id
                                                                    , query_text = queryQuestions.query_text
                                                                    , unanswerable_expressions = option_non_answers
                                                                    )
            elif(prompt_class == "QuestionAnswerablePromptWithChoices"):
                qpc = QuestionAnswerablePromptWithChoices(question_id = question.question_id
                                                                    , question = question.question_text
                                                                    , query_id = question.query_id
                                                                    , facet_id = question.facet_id
                                                                    , query_text = queryQuestions.query_text
                                                                    , unanswerable_expressions = option_non_answers
                                                                    )
            else:
                raise RuntimeError(f"Prompt class {prompt_class} not supported by naghmeh's question_loader.")\
        

            qpc_dict[query_id].append(qpc)
    return list(qpc_dict.items())

## -------------------------------------------        

def main():
    question1 = ExamQuestion(question_id="12897q981", query_id="Q1", question_text="What was my question, again?", facet_id=None, info=None)
    question2 = ExamQuestion(question_id="42", query_id="Q1", question_text="Who am I?", facet_id="some_facet", info=None)

    print(pydantic_dump(question1))
    queryBank = QueryQuestionBank(query_id="Q1", facet_id=None, test_collection="dummy", query_text="everything", info=None
                      , questions= [question1, question2]
                      )


    writeQuestionBank("newfile.jsonl.gz", [queryBank, queryBank])

    bank_again = parseQuestionBank("newfile.jsonl.gz")
    print(bank_again[0])

    qpcs = load_exam_question_bank("newfile.jsonl.gz", prompt_class="QuestionSelfRatedUnanswerablePromptWithChoices")
    print(qpcs[0])

if __name__ == "__main__":
    main()

