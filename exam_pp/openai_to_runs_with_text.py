import datetime
import os
import random
from typing import Optional
from openai import OpenAI
import openai
import requests
import time
import datetime

import trec_car.read_data as trec_car
from .davinci_to_runs_with_text import *

    # # Parse rate limit information
    # rate_limit_info = {
    #     'limit_requests': headers.get('x-ratelimit-limit-requests'),
    #     'limit_tokens': headers.get('x-ratelimit-limit-tokens'),
    #     'remaining_requests': headers.get('x-ratelimit-remaining-requests'),
    #     'remaining_tokens': headers.get('x-ratelimit-remaining-tokens'),
    #     'reset_requests': headers.get('x-ratelimit-reset-requests'),
    #     'reset_tokens': headers.get('x-ratelimit-reset-tokens')
    # }


if os.environ['OPENAI_API_KEY'] is None:
    raise RuntimeError ("Must set environment variable \"OPENAI_API_KEY\"")

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])





class OpenAIRateLimiter:
    def __init__(self, max_requests_per_minute=5000, max_tokens_per_minute=40000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.remaining_requests = max_requests_per_minute
        self.remaining_tokens = max_tokens_per_minute
        self.start_time = time.time()

    def wait_if_needed(self):
        if self.remaining_requests <= 0 or self.remaining_tokens <= 0:
            time.sleep(max(60 - (time.time() - self.start_time), 0))
            self.remaining_requests = self.max_requests_per_minute
            self.remaining_tokens = self.max_tokens_per_minute
            self.start_time = time.time()

    def update_limits(self, used_tokens):
        self.remaining_requests -= 1
        self.remaining_tokens -= used_tokens

    def __str__(self) -> str:
        if self.start_time:
            fmt_time = time.strftime("%H:%M:%S",time.localtime(self.start_time))
            return f'remaining_request={self.remaining_requests}  remaining_tokens={self.remaining_tokens}  start_time={fmt_time}'
        return ""

# Initialize rate limiter
rate_limiter = OpenAIRateLimiter()

 
# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""
 
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
 
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
 
            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1
 
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
 
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
 
                # Sleep for the delay
                time.sleep(delay)
 
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
 
    return wrapper

def query_gpt_batch_with_rate_limiting(prompt:str, rate_limiter, gpt_model:str):
    result = []

    rate_limiter.wait_if_needed()

    messages = [{"role":"user", "content":prompt}]
    # messaes = [openai.ChatCompletionUserMessageParam(content)]
    completion = retry_with_exponential_backoff(func=client.chat.completions.create)(model=gpt_model,messages=messages, max_tokens=1500) 
    # print(completion.model_dump_json(indent=2))

    result = [choice.message.content.strip() for choice in completion.choices]

    # Update rate limits
    usage = dict(completion).get('usage')

    print("usage", usage)
    if usage is not None:
        used_tokens = dict(usage).get('total_tokens')
        rate_limiter.update_limits(used_tokens)
    else:
        raise RuntimeError("usage not provided")

    print(rate_limiter)
    return result[0]



class FetchGptResponsesForTrecCar:
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter

    def section_prompt(self, query_title:str, query_heading:str)->str:
        return f"Generate a Wikipedia section on \"{query_heading}\" for an article on \"{query_title}\"."

    def page_prompt(self, query_title:str)->str:
        return f"Generate a 1000-word long Wikipedia article on \"{query_title}\"."

    def generate(self, prompt:str, gpt_model:str)->str:
        answer = query_gpt_batch_with_rate_limiting(prompt,rate_limiter=self.rate_limiter, gpt_model=gpt_model)
        return answer


def noodle_gpt(page_davinci_path:Path, gpt_out:Path, gpt_model:str):
    fetcher = FetchGptResponsesForTrecCar(OpenAIRateLimiter())

    davinci_by_query_id = parse_davinci_into_dict(section_file_path=None, page_file_path=page_davinci_path)

    with open(gpt_out, "wt", encoding='utf-8') as file:
        for query_id, davincis in davinci_by_query_id.items(): # itertools.islice(davinci_by_query_id.items(),1):
            for davinci in davincis:
                answer = fetcher.generate(davinci.prompt, gpt_model=gpt_model)
                answer = fetcher.generate(davinci.prompt, gpt_model=gpt_model)
                davinci.response=answer
                davinci.gptmodel=gpt_model

                davinci.datatime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                file.write(pydantic_dump(davinci)+'\n')
                print(query_id, rate_limiter)
            file.flush()
        file.close()





def main():
    # rate_limiter=OpenAIRateLimiter()
    # fetcher = FetchGptResponsesForTrecCar(rate_limiter=rate_limiter)
    # answer = fetcher.generate(["the skeletal system"])
    # print(answer)

    page_davinci_path = "./v24-20-lucene-page--text-davinci-003-benchmarkY3test.jsonl"
    gpt_model="gpt-3.5-turbo"
    gpt_out = Path(f"./openai-{gpt_model}-benchmarkY3test.jsonl")

    noodle_gpt(page_davinci_path=page_davinci_path, gpt_model=gpt_model, gpt_out=gpt_out)


if __name__ == "__main__":
    main()
