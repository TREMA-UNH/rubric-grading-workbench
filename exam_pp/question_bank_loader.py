from collections import defaultdict
import itertools
import typing
from pydantic import BaseModel
import json
from typing import List, Any, Optional, Dict, Set, Tuple
from dataclasses import dataclass
import gzip
from pathlib import Path

from .question_types import QuestionAnswerablePromptWithChoices, QuestionCompleteConciseUnanswerablePromptWithChoices, QuestionPrompt, QuestionSelfRatedUnanswerablePromptWithChoices


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
        file.writelines([(questionBank.json()+'\n') for questionBank in queryQuestionBanks])

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

    print(question1.json())
    queryBank = QueryQuestionBank(query_id="Q1", facet_id=None, test_collection="dummy", query_text="everything", info=None
                      , questions= [question1, question2]
                      )


    writeQuestionBank("newfile.json.gz", [queryBank, queryBank])

    bank_again = parseQuestionBank("newfile.json.gz")
    print(bank_again[0])

    qpcs = load_exam_question_bank("newfile.json.gz", prompt_class="QuestionSelfRatedUnanswerablePromptWithChoices")
    print(qpcs[0])

if __name__ == "__main__":
    main()

