import typing
from pydantic import BaseModel
import json
from typing import List, Any, Optional, Dict, Tuple
from dataclasses import dataclass
import gzip
from pathlib import Path


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



def main():
    question1 = ExamQuestion(question_id="12897q981", query_id="Q1", question_text="What was my question, again?", facet_id=None, info=None)
    question2 = ExamQuestion(question_id="42", query_id="Q1", question_text="Who am I?", facet_id="some_facet", info=None)

    print(question1.json())
    bank = QueryQuestionBank(query_id="Q1", facet_id=None, test_collection="dummy", query_text="everything", info=None
                      , questions= [question1, question2]
                      )


    writeQuestionBank("newfile.json.gz", [bank])

    bank_again = parseQuestionBank("newfile.json.gz")
    print(bank_again[0])


if __name__ == "__main__":
    main()

