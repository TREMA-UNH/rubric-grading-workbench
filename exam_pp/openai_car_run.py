import time
import datetime

import trec_car.read_data as trec_car

from .openai_interface import query_gpt_batch_with_rate_limiting
from .davinci_to_runs_with_text import *


class FetchGptResponsesForTrecCar:
    def __init__(self):
        pass

    def section_prompt(self, query_title:str, query_heading:str)->str:
        return f"Generate a Wikipedia section on \"{query_heading}\" for an article on \"{query_title}\"."

    def page_prompt(self, query_title:str)->str:
        return f"Generate a 1000-word long Wikipedia article on \"{query_title}\"."

    def generate(self, prompt:str, gpt_model:str,max_tokens:int)->str:
        answer = query_gpt_batch_with_rate_limiting(prompt, gpt_model=gpt_model, max_tokens=max_tokens)
        return answer


def noodle_from_prior_prompts(page_davinci_path:Path, gpt_out:Path, gpt_model:str, max_tokens:int=1500):
    fetcher = FetchGptResponsesForTrecCar()

    davinci_by_query_id = parse_davinci_into_dict(section_file_path=None, page_file_path=page_davinci_path)

    with open(gpt_out, "wt", encoding='utf-8') as file:
        for query_id, davincis in davinci_by_query_id.items(): # itertools.islice(davinci_by_query_id.items(),1):
            for davinci in davincis:
                # answer = fetcher.generate(davinci.prompt, gpt_model=gpt_model)
                answer = fetcher.generate(davinci.prompt, gpt_model=gpt_model, max_tokens=max_tokens)
                davinci.response=answer
                davinci.gptmodel=gpt_model

                davinci.datatime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                file.write(pydantic_dump(davinci)+'\n')
                print(query_id)
            file.flush()
        file.close()



def noodle_cary3_outlines(cary3_outlines:Path, gpt_out:Path, gpt_model:str, max_tokens:int=1500):
    fetcher = FetchGptResponsesForTrecCar()
    with open(gpt_out, "wt", encoding='utf-8') as file:
        page:trec_car.Page
        for page in trec_car.iter_outlines(open(cary3_outlines, 'rb')):
            query_id = page.page_id
            query_text = page.page_name

            prompt = fetcher.page_prompt(query_text)
            answer = fetcher.generate(prompt, gpt_model=gpt_model, max_tokens=max_tokens)
            datatime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            davinci = DavinciResponse(datatime=datatime
                                        , response=answer
                                        , gptmodel=gpt_model
                                        , prompt=prompt
                                        , benchmark="cary3"
                                        , queryId=query_id
                                        , queryStr=query_text
                                        , sectionQueryStr=None
                                        , pageOrSection="page")

            file.write(pydantic_dump(davinci)+'\n')
            print(query_id)
            file.flush()
        file.close()





def main():
    import argparse

    desc = r'''Create CAR-Y3 content from OpenAI'''
    
    parser = argparse.ArgumentParser(description="Create Wiki articles with ChatGPT"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-p','--davinci-page-file', type=str, metavar='DAVINCI_PAGE_FILE'
                        , help='Input jsonl file for page-level Davinci3 responses'
                        )
    parser.add_argument('-c','--car-outlines-cbor', type=str, metavar='CAR_OUTLINES_CBOR', required=True
                        , help='Input TREC CAR ourlines file (from which page-level queries and order of sections/facets will be taken)'
                        )


    parser.add_argument('-o', '--out-file', type=str, metavar='runs-xxx.jsonl.gz', required=True
                        , help='Output file name where paragraphs with exam grade annotations will be written to')

    parser.add_argument('--gpt-model', type=str, metavar='MODEL', default="gpt-3.5-turbo", help='OpenAI model name to be used')
    parser.add_argument('--max-tokens', type=int, metavar='NUM', default=1500, help='Max Tokens from OpenAI')
 
    args = parser.parse_args()  


    if args.davinci_page_file is not None:
        noodle_from_prior_prompts(page_davinci_path=args.davinci_page_file, gpt_model=args.gpt_model, gpt_out=args.out_file, max_tokens=args.max_tokens)
    
    elif args.car_outlines_cbor is not None:
        noodle_cary3_outlines(cary3_outlines=args.car_outlines_cbor, gpt_model=args.gpt_model, gpt_out=args.out_file, max_tokens=args.max_tokens)
    else:
        raise RuntimeError("Must either define --davinci-page-file or --car-outlines-cbor !")

if __name__ == "__main__":
    main()
