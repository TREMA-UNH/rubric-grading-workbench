import pandas as pd
import numpy as np
from typing import NewType, List, Optional, Dict, Any
from pathlib import Path
from enum import Enum, auto


import duckdb

from .data_model import *


RelevanceItemId = NewType('RelevanceItemId', int)
ExamGradeId = NewType('ExamGradeId', int)




SCHEMA = '''--sql
-- What is the content of the paragraph generated/retrieved for the query
-- queryId, paragraph_id 
CREATE SEQUENCE relevance_item_id_seq START 1;
CREATE TABLE relevance_item (
    relevance_item_id INTEGER PRIMARY KEY DEFAULT nextval('relevance_item_id_seq'),
    query_id text,
    paragraph_id text,
    paragraph_text text,
    metadata JSON,
);


-- -- Who ranked this item and how high?
-- -- relevance_item_id -> method:str, rank:int, score:float
-- CREATE SEQUENCE ranked_item_id_seq START 1;
-- CREATE TABLE ranked_item (
--     ranked_item_id INTEGER PRIMARY KEY DEFAULT nextval('ranked_item_id_seq'),
--     metadata JSON,
-- );

-- -- Is this item relevant according to human/true relevance judgments?
-- -- relevance_item_id -> relevance_label



-- Exam_grades: Rubric grading information
-- (relevance_item_id, test_bank_id/question_id) 
--      -> (prompt_class, prompt_type, llm, llm_options, prompt_info) 
--      -> (self_rating:int, correct:bool, answer:str)
CREATE SEQUENCE exam_grade_id_seq START 1;
CREATE TABLE exam_grade (
    exam_grade_id INTEGER PRIMARY KEY DEFAULT nextval('exam_grade_id_seq'),
    relevance_item_id integer references relevance_item(relevance_item_id),
    test_bank_id text,  --todo the test banks should also go into a table
    test_bank_type text,
    test_bank_metadata JSON,
-- 
    prompt_class text,
    prompt_type text,
    prompt_is_self_rated boolean,
    prompt_info JSON,
    llm text,
    llm_options JSON,
-- 
    self_rating integer,
    is_correct boolean,
    answer text,
    metadata JSON,
);

CREATE SEQUENCE exam_grade_error_id_seq START 1;
CREATE TABLE exam_grade_errors (
    exam_grade_error_id INTEGER PRIMARY KEY DEFAULT nextval('exam_grade_error_id_seq'),
    relevance_item_id integer references relevance_item(relevance_item_id),
    test_bank_id text,  --todo the test banks should also go into a table
    test_bank_type text,
    test_bank_metadata JSON,
-- 
    prompt_class text,
    prompt_type text,
    prompt_is_self_rated boolean,
    prompt_info JSON,
    llm text,
    llm_options JSON,
-- 
    llm_response_error JSON,
    metadata JSON,
);
'''




class RubricDb:
    def __init__(self, path: Path):
        needs_init = not path.exists()
        self.db = duckdb.connect(path)
        if needs_init:
            print(f'initializing {path}')
            self.db.begin()
            self.db.execute(SCHEMA)
            self.db.commit()


def import_rubric_data(data_path:Path, rubric_db:RubricDb):
    data = parseQueryWithFullParagraphs(data_path)
    for qp in data:
        query_id = qp.queryId
        for para in qp.paragraphs:
            rubric_db.db.execute(
                '''--sql
                INSERT INTO relevance_item (query_id, paragraph_id, paragraph_text, metadata)
                VALUES (?, ?, ?, ?)
                RETURNING relevance_item_id
                ''',
                (query_id, para.paragraph_id, para.text, para.paragraph)
            )
            relevance_item_id, = rubric_db.db.fetchone()

            if para.exam_grades is not None:
                for exam_grades in para.exam_grades:

                    # exam_grade_meta = GradeFilter.key_dict(exam_grades)
                    prompt_class = GradeFilter.get_prompt_class(exam_grades)
                    prompt_type = GradeFilter.get_prompt_type(exam_grades)
                    is_self_rated = GradeFilter.get_is_self_rated(exam_grades)
                    test_bank_type = GradeFilter.get_question_type(exam_grades)
                    
                    llm = exam_grades.llm
                    llm_meta = exam_grades.llm_options
                    prompt_meta = exam_grades.prompt_info

                    answers = {test_bank_id:answer for test_bank_id, answer in exam_grades.answers}
                    correct_ids = set(exam_grades.correctAnswered)
                    wrong_ids = set(exam_grades.wrongAnswered)
                    self_ratings = {}
                    if exam_grades.self_ratings is not None:
                        self_ratings = {r.get_id(): r.self_rating for r in exam_grades.self_ratings}

                    all_test_bank_ids = correct_ids.union(wrong_ids).union(answers.keys()).union(self_ratings.keys())

                    for test_bank_id in all_test_bank_ids:
                        answer = answers.get(test_bank_id)
                        correctAnswered = test_bank_id in correct_ids
                        self_rating = self_ratings.get(test_bank_id)

                        # submit!
                        # print(f"{query_id}, {paragraph_id}, {prompt_class}, {llm}, {test_bank_id}, {answer}, {correctAnswered}, {self_rating}")
                        rubric_db.db.execute(
                            '''--sql
                            INSERT INTO exam_grade (
                                relevance_item_id, test_bank_id, test_bank_type, test_bank_metadata,
                                prompt_class, prompt_type, prompt_is_self_rated, prompt_info, llm, llm_options,
                                self_rating, is_correct, answer, metadata )
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                            ''',
                            (relevance_item_id,test_bank_id,test_bank_type, None
                             , prompt_class, prompt_type, is_self_rated, prompt_meta, llm, llm_meta
                             , self_rating, correctAnswered, answer, None
                             )
                        )
                    
                    if exam_grades.llm_response_errors is not None:
                        # put LLM response errors in table
                        for test_bank_id, llm_err in exam_grades.llm_response_errors.items():

                            rubric_db.db.execute(
                                '''--sql
                                INSERT INTO exam_grade_errors (
                                    relevance_item_id, test_bank_id, test_bank_type, test_bank_metadata,
                                    prompt_class, prompt_type, prompt_is_self_rated, prompt_info, llm, llm_options,
                                    self_rating, is_correct, answer, metadata )
                                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                                ''',
                                (relevance_item_id,test_bank_id,test_bank_type, None
                                , prompt_class, prompt_type, is_self_rated, prompt_meta, llm, llm_meta
                                , llm_err, None
                                )
                            )

#     exam_grade_id INTEGER PRIMARY KEY DEFAULT nextval('exam_grade_id_seq'),
#     relevance_item_id integer references relevance_item(relevance_item_id),
#     test_bank_id integer,  --todo the test banks should also go into a table
#     test_bank_type text
#     test_bank_metadata JSON,
# -- 
#     prompt_class text,
#     prompt_type text,
#     prompt_is_self_rated boolean,
#     prompt_info JSON,
#     llm text,
#     llm_options JSON,
# -- 
#     self_rating integer,
#     is_correct boolean,
#     answer text,
#     metadata JSON,

def main_rubric_db():

    import argparse

    desc = f'''EXAM convert RUBRIC data from jsonl.gz to duckdb. 
             '''
    
    parser = argparse.ArgumentParser(description="EXAM convert to DuckDB"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter
                                   )
    parser.add_argument('paragraph_file', type=str, metavar='xxx.jsonl.gz'
                        , help='json file with paragraph to grade with exam questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )


    parser.add_argument('-o', '--out-db', type=str, metavar='exam-xxx.rubric.duckdb', help='Export duckdb file (file must not exist!). By default will use paragraph_file, but with extension rubric.duckdb instead of jsonl.gz')
    # parser.add_argument('--question-path', type=str, metavar='PATH', help='Path to read grading rubric exam questions/nuggets from (can be tqa directory or file)')
    # parser.add_argument('--use-nuggets', action='store_true', help="if set, assumed --question-path contains nuggets instead of questions")
    # parser.add_argument('--question-type', type=str, choices=['question-bank','direct', 'tqa','genq'], default="question-bank", metavar='PATH', help='Grading rubric file format for reading from --question-path')


    # parser.add_argument('--max-queries', type=int, metavar="n", default=-1, help="Limit number of queries to be processed")
    # parser.add_argument('--max-paragraphs', type=int, metavar="n", default=-1, help="Limit number of paragraphs to be processed")


    args = parser.parse_args()

    rubric_paragraph_file = Path(args.paragraph_file)
    rubric_db_path = args.out_db or Path(args.paragraph_file.replace(".jsonl.gz", ".rubric.duckdb"))

    rubric_db = RubricDb(Path(rubric_db_path))
    rubric_db.db.begin()
    import_rubric_data(Path(rubric_paragraph_file), rubric_db)
    rubric_db.db.commit()



    print("relevance_item", rubric_db.db.execute('''
                        SELECT COUNT(*) FROM relevance_item
                         ''').fetchone()    )

    print("exam_grade", rubric_db.db.execute('''
                        SELECT COUNT(*) FROM exam_grade
                         ''').fetchone()    )

    print("exam_grade_errors", rubric_db.db.execute('''
                        SELECT COUNT(*) FROM exam_grade_errors
                         ''').fetchone()    )


if __name__ == "__main__":
    main_rubric_db()

