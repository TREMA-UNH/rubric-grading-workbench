from io import TextIOWrapper
from json import JSONDecodeError
import os
import time
import datetime
from typing import TextIO

import trec_car.read_data as trec_car

from .question_bank_loader import ExamQuestion, QueryQuestionBank, emit_exam_question_bank_entry, write_single_query_test_bank, writeTestBank

from .openai_interface import query_gpt_batch_with_rate_limiting
from .davinci_to_runs_with_text import *


class FetchGptJson:

    def __init__(self, gpt_model:str, max_tokens:int):
        self.gpt_model = gpt_model
        self.max_tokens = max_tokens


    def set_json_instruction(self, json_instruction:str, field_name:str):
        self._json_instruction = json_instruction
        self._field_name = field_name


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
            list = response.get(self._field_name)
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
        full_prompt = prompt+self._json_instruction

        print("\n\n"+full_prompt+"\n\n")

        tries = 3
        while tries>0:
            response = self._generate( prompt=full_prompt
                                      , gpt_model=self.gpt_model
                                      , max_tokens=self.max_tokens)
            questions = self._parse_json_response(response)
            if questions is not None:
                return questions
            else:
                tries-=1
                print(f"Receiving unparsable response: {response}. Tries left: {tries}")
        return None

            

class FetchGptQuestions(FetchGptJson):
    def __init__(self, gpt_model:str, max_tokens:int):
        super().__init__(gpt_model=gpt_model, max_tokens=max_tokens)
        json_instruction= r'''
Give the question set in the following JSON format:
```json
{ "questions": [question_text_1, question_text_2, ... ] }
```'''
        field_name="questions"
        self.set_json_instruction(json_instruction=json_instruction, field_name=field_name)


class FetchGptNuggets(FetchGptJson):
    def __init__(self, gpt_model:str, max_tokens:int):
        super().__init__(gpt_model=gpt_model, max_tokens=max_tokens)
        json_instruction= r'''
Give the nugget set in the following JSON format:
```json
{ "nuggets": [nugget_text_1, nugget_text2, ... ] }
```'''
        field_name="nuggets"
        self.set_json_instruction(json_instruction=json_instruction, field_name=field_name)






#
#        Nuggets are stored in  
    #         {
    #   "question_id": "tqa2:L_0002/T_0018/bdde682b973b1faa0c73d5e54deabcf7",
    #   "question_text": "Troposphere houses weather phenomena",
# 









# ---------------- CAR Y3 -------------------

def car_section_question_prompt(query_title:str, query_subtopic:str)->str:
    return f'''Explore the connection between '{query_title}' with a specific focus on the subtopic '{query_subtopic}'.
        Generate insightful questions that delve into advanced aspects of '{query_subtopic}', showcasing a deep understanding of the subject matter. Avoid basic or introductory-level inquiries. '''
    
def car_section_nugget_prompt(query_title:str, query_subtopic:str)->str:
    return f'''Explore the connection between '{query_title}' with a specific focus on the subtopic '{query_subtopic}'.
        Generate insightful nuggets (key facts) that delve into advanced aspects of '{query_subtopic}', showcasing a deep understanding of the subject matter. Avoid basic or introductory-level nuggets. Keep nuggets to a maximum of 4 words.'''    


def generate_questions_cary3(car_outlines_path:Path, fetcher: FetchGptJson, out_file:TextIO, use_nuggets:bool, max_queries:Optional[int]=None, test_collection:str="cary3"):
   
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
            prompt =""
            if use_nuggets:
                prompt = car_section_nugget_prompt(query_title=title_query, query_subtopic=section_query)
            else:
                prompt = car_section_question_prompt(query_title=title_query, query_subtopic=section_query)


            questions = fetcher.generate_question(prompt)
            emit_exam_question_bank_entry(out_file, test_collection, generation_info, query_id, query_facet_id, query_text, questions)


#  ------------------- TREC DL ------------------



def dl_query_prompt(query_text:str)->str:
    return f'''Break the query '{query_text}' into concise questions that must be answered. 
     Generate 10 concise insightful questions that reveal whether information relevant for '{query_text}' was provided, showcasing a deep understanding of the subject matter. Avoid basic or introductory-level inquiries. Keep the questions short.'''



def generate_questions_dl(query_json:Path, fetcher: FetchGptQuestions, out_file:TextIO, max_queries:Optional[int]=None, test_collection:str="cary3"):
  
    generation_info = fetcher.generation_info()
    
    queries:Dict[str,str]
    with open(query_json, "rt", encoding="utf-8") as file:
        queries = json.load(file)

        for query_id, query_text in itertools.islice(queries.items(), max_queries):
            print(query_id, query_text)
            prompt = dl_query_prompt(query_text=query_text)

            questions = fetcher.generate_question(prompt)
            emit_exam_question_bank_entry(out_file, test_collection, generation_info, query_id, None, query_text, questions)

# -------------------------------------------
            

def noodle_car_gpt(car_outlines_cbor:Path, gpt_out:Path, gpt_model:str, use_nuggets:bool, max_tokens:int=1500):
    json_instruction= r'''
Give the question set in the following JSON format:
```json
{ "questions": [question_text_1, question_text_2, ... ] }
```'''
    field_name="questions"



    fetcher:FetchGptJson

    fetcher = FetchGptQuestions(gpt_model=gpt_model, max_tokens=max_tokens)
    if use_nuggets:
        fetcher = FetchGptNuggets(gpt_model=gpt_model, max_tokens=max_tokens)


    with gzip.open(gpt_out, "wt", encoding='utf-8') as file:
        generate_questions_cary3(car_outlines_path = car_outlines_cbor, fetcher=fetcher, out_file = file, use_nuggets=use_nuggets)

        file.close()

def noodle_query_gpt(query_json:Path, gpt_out:Path, gpt_model:str, max_tokens:int=1500):
    fetcher = FetchGptQuestions(gpt_model=gpt_model, max_tokens=max_tokens)

    with gzip.open(gpt_out, "wt", encoding='utf-8') as file:
        generate_questions_dl(query_json = query_json, fetcher=fetcher, out_file = file)

        file.close()





def main():
    os.environ['OPENAI_API_KEY']="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"

    import argparse

    query_json_format=r'''
        {
          "query_id_1":"query text",
          "query_id_2":"query text",
          "query_id_3":"query text",
          ...
        }
'''
    desc = r'''Generate Questions for CAR Queries'''
    
    parser = argparse.ArgumentParser(description="Question Generation"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-c','--car-outlines-cbor', type=str, metavar='CAR_OUTLINES_CBOR'
                        , help='Input TREC CAR ourlines file (from which page-level queries and order of sections/facets will be taken)'
                        )
    
    parser.add_argument('-q','--query-json', type=str, metavar='JSON'
                        , help=f'Input query-file in JSON format: {query_json_format}'
                        )
    
    parser.add_argument('--use-nuggets', action='store_true', help="if set uses nuggets instead of questions")


    parser.add_argument('-o', '--out-file', type=str, metavar='runs-xxx.jsonl.gz', required=True
                        , help='Output file name where paragraphs with exam grade annotations will be written to')

    parser.add_argument('--gpt-model', type=str, metavar='MODEL', default="gpt-3.5-turbo", help='OpenAI model name to be used')
    parser.add_argument('--max-tokens', type=int, metavar='NUM', default=1500, help='Max Tokens from OpenAI')
 
    args = parser.parse_args()  

    if not (args.car_outlines_cbor  or args.query_json):
        raise RuntimeError("Either --car-outines-cbor or --query-json must be set.")
    if  (args.car_outlines_cbor  and args.query_json):
        raise RuntimeError("Either --car-outines-cbor or --query-json must be set. (not both)")

    if args.car_outlines_cbor is not None:
        noodle_car_gpt(car_outlines_cbor=args.car_outlines_cbor, gpt_model=args.gpt_model, gpt_out=args.out_file, max_tokens=args.max_tokens, use_nuggets = args.use_nuggets)

    if args.query_json is not None:
        noodle_query_gpt(query_json=args.query_json, gpt_model=args.gpt_model, gpt_out=args.out_file, max_tokens=args.max_tokens)

if __name__ == "__main__":
    main()
