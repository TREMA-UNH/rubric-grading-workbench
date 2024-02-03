from io import TextIOWrapper
from json import JSONDecodeError
import time
import datetime
from typing import TextIO

import trec_car.read_data as trec_car

from .question_bank_loader import ExamQuestion, QueryQuestionBank, emit_exam_question_bank_entry, write_single_query_question_bank, writeQuestionBank

from .openai_interface import query_gpt_batch_with_rate_limiting
from .davinci_to_runs_with_text import *


class FetchGptQuestions:

    def __init__(self, gpt_model:str, max_tokens:int):
        self.gpt_model = gpt_model
        self.max_tokens = max_tokens
        self.json_instruction = r'''
Give the question set in the following JSON format:
```json
{ "questions": [question_text_1, question_text2, ... ] }
```'''

    def generation_info(self):
        return {"gpt_model":self.gpt_model
                , "format_instruction":"json"
                , "prompt_style": "Explore the connection..."
                }


    def __is_list_of_strings(self, lst):
        return isinstance(lst, list) and all(isinstance(item, str) for item in lst)

    def _parse_json_response(self, gpt_response:str) -> Optional[List[str]]:

        try:
            response = json.loads(gpt_response)
            list = response.get("questions")
            if(self.__is_list_of_strings(list)) is not None:
                return list
            else:
                return None
        except JSONDecodeError as e:
            print(e)
            return None

    def _generate(self, prompt:str, gpt_model:str,max_tokens:int)->str:
        answer = query_gpt_batch_with_rate_limiting(prompt, gpt_model=gpt_model, max_tokens=max_tokens)
        return answer

    def generate_question(self, prompt:str):

        tries = 3
        while tries>0:
            response = self._generate( prompt=prompt+self.json_instruction
                                      , gpt_model=self.gpt_model
                                      , max_tokens=self.max_tokens)
            questions = self._parse_json_response(response)
            if questions is not None:
                return questions
            else:
                tries-=1
                print(f"Receiving unparsable response: {response}. Tries left: {tries}")
        return None
            


def car_section_prompt(query_title:str, query_subtopic:str)->str:
    # return f"Generate a Wikipedia section on \"{query_heading}\" for an article on \"{query_title}\"."
    return f'''Explore the connection between '{query_title}' with a specific focus on the subtopic '{query_subtopic}'.
        Generate insightful questions that delve into advanced aspects of '{query_subtopic}', showcasing a deep understanding of the subject matter. Avoid basic or introductory-level inquiries. '''
    


def parse(car_outlines_path:Path, use_facets:bool, fetcher: FetchGptQuestions, out_file:TextIO, max_queries:Optional[int]=None, test_collection:str="cary3"):

    generation_info = fetcher.generation_info()
    
    page:trec_car.Page
    for page in itertools.islice(trec_car.iter_outlines(open(car_outlines_path, 'rb')), max_queries):

        query_id = page.page_id
        title_query = page.page_name

        section:trec_car.Section
        for section in page.child_sections:
            query_facet_id = section.headingId
            section_query = section.heading
            query_text = f'{title_query} / {section_query}'


            print(query_id, query_facet_id, title_query, section_query)
            prompt = car_section_prompt(query_title=title_query, query_subtopic=section_query)

            questions = fetcher.generate_question(prompt)

            emit_exam_question_bank_entry(use_facets, out_file, test_collection, generation_info, query_id, query_facet_id, query_text, questions)




def noodle_gpt(car_outlines_cbor:Path, gpt_out:Path, gpt_model:str, max_tokens:int=1500):
    fetcher = FetchGptQuestions(gpt_model=gpt_model, max_tokens=max_tokens)

    with gzip.open(gpt_out, "wt", encoding='utf-8') as file:
        parse(car_outlines_path = car_outlines_cbor, use_facets=True, fetcher=fetcher, out_file = file)

        file.close()





def main():
    import argparse

    desc = r'''Generate Questions for CAR Queries'''
    
    parser = argparse.ArgumentParser(description="Question Generation"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-c','--car-outlines-cbor', type=str, metavar='CAR_OUTLINES_CBOR', required=True
                        , help='Input TREC CAR ourlines file (from which page-level queries and order of sections/facets will be taken)'
                        )


    parser.add_argument('-o', '--out-file', type=str, metavar='runs-xxx.jsonl.gz', required=True
                        , help='Output file name where paragraphs with exam grade annotations will be written to')

    parser.add_argument('--gpt-model', type=str, metavar='MODEL', default="gpt-3.5-turbo", help='OpenAI model name to be used')
    parser.add_argument('--max-tokens', type=int, metavar='NUM', default=1500, help='Max Tokens from OpenAI')
 
    args = parser.parse_args()  


    noodle_gpt(car_outlines_cbor=args.car_outlines_cbor, gpt_model=args.gpt_model, gpt_out=args.out_file, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
