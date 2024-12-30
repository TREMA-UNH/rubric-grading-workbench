import datetime
from json import JSONDecodeError
import os
import random
from typing import Optional
from httpx import URL
from openai import OpenAI
import openai
import requests
import time
import datetime

import trec_car.read_data as trec_car

from .exam_llm import LlmResponseError
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




def createOpenAIClient(api_key:str|None=os.getenv('OPENAI_API_KEY'), base_url:str|URL|None=None):
    if api_key is None:
        raise RuntimeError ("api_key must either be set as argument or via environment variable \"OPENAI_API_KEY\"")

    return OpenAI(api_key=api_key,base_url=base_url)

# client = createOpenAIClient()

_client:Optional[OpenAI] = None

def default_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = createOpenAIClient()
    return _client

class OpenAIRateLimiter:
    def __init__(self, max_requests_per_minute=5000, max_tokens_per_minute=40000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.remaining_requests = max_requests_per_minute
        self.remaining_tokens = max_tokens_per_minute
        self.start_time = time.time()
        

    def wait_if_needed(self):
        if self.remaining_requests <= 0 or self.remaining_tokens <= 0:
            wait_secs = max(60 - (time.time() - self.start_time),0)
            if wait_secs > 0:
                print(f"OpenAIRateLimiter waiting {wait_secs} seconds. {self}")
                time.sleep(wait_secs)
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
global_rate_limiter = OpenAIRateLimiter()

 
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

def query_gpt_batch_with_rate_limiting(prompt:str, gpt_model:str, max_tokens:int, use_chat_interface:bool, client=None,  rate_limiter:Optional[OpenAIRateLimiter]=None, system_message:Optional[str]=None, **kwargs):
    if client is None:
        client = default_openai_client()
    if rate_limiter is None:
        rate_limiter = global_rate_limiter

    result = []

    rate_limiter.wait_if_needed()


    if use_chat_interface:
        messages = list()
        if system_message is not None:
            messages.append({"role":"system", "content":system_message})
        messages.append({"role":"user", "content":prompt})
        completion = retry_with_exponential_backoff(func=client.chat.completions.create)(model=gpt_model,messages=messages, max_tokens=max_tokens, **kwargs) 
        result = [choice.message.content.strip() for choice in completion.choices]
    else:
        completion = retry_with_exponential_backoff(func=client.completions.create)(model=gpt_model,prompt=prompt, max_tokens=max_tokens, **kwargs) 
        result = [choice.text.strip() for choice in completion.choices]


    # Update rate limits
    usage = dict(completion).get('usage')

    # print("usage", usage)
    if usage is not None:
        used_tokens = dict(usage).get('total_tokens')
        rate_limiter.update_limits(used_tokens)
    else:
        raise RuntimeError("usage not provided")

    # print(rate_limiter)
    return result[0]





class FetchGptJson:
    def __init__(self, gpt_model:str, max_tokens:int, client=None, use_chat_protocol:bool=True):
        self.client = client if client is not None else default_openai_client()
        self.gpt_model = gpt_model
        self.max_tokens = max_tokens
        self.use_chat_protocol = use_chat_protocol


    def set_json_instruction(self, json_instruction:str, field_name:str):
        self._json_instruction = json_instruction
        self._field_name = field_name


    def generation_info(self):
        return {"gpt_model":self.gpt_model
                , "format_instruction":"json"
                , "prompt_target": self._field_name
                }


    def __is_list_of_strings(self, lst):
        return isinstance(lst, list) and all(isinstance(item, str) for item in lst)

    def __is_int(self, i):
        return isinstance(i,int)

    def _parse_json_response(self, gpt_response:str, request:Optional[str]=None) -> Optional[str]:
        resp = gpt_response.strip()
        cleaned_gpt_response=""
        if resp.startswith("{"):
            cleaned_gpt_response=resp
        elif resp.startswith("```json"):
            cleaned_gpt_response= re.sub(r'```json|```', '', resp).strip()
        elif resp.startswith("```"):
            cleaned_gpt_response= re.sub(r'```|```', '', resp).strip()
        else:
            print(f"Not sure how to parse ChatGPT response from json:\n-----\n{request}\n-----\n{gpt_response}\n----")
            cleaned_gpt_response=resp

        try:
            response = json.loads(cleaned_gpt_response)
            grade = response.get(self._field_name)
            if(self.__is_int(grade)) is not None:
                return f"{grade}"
            else:
                return None
        except JSONDecodeError as e:
            print(e)
            return None

    def _generate(self, prompt:str, gpt_model:str,max_tokens:int, rate_limiter:OpenAIRateLimiter, system_message:Optional[str]=None, **kwargs)->str:
        answer = query_gpt_batch_with_rate_limiting( prompt, gpt_model=gpt_model, max_tokens=max_tokens, client=self.client, rate_limiter=rate_limiter, use_chat_interface=self.use_chat_protocol, system_message=system_message, **kwargs)
        return answer

    async def generate_request(self, prompt:str, rate_limiter:OpenAIRateLimiter, system_message:Optional[str]=None, **kwargs)->Union[str, LlmResponseError]:
        full_prompt = prompt+self._json_instruction

        # print("\n\n"+full_prompt+"\n\n")

        tries = 3
        while tries>0:
            try:
                response = self._generate( prompt=full_prompt
                                        , gpt_model=self.gpt_model
                                        , max_tokens=self.max_tokens
                                        , rate_limiter=rate_limiter
                                        , system_message=system_message
                                        , **kwargs
                                        )
                # print(response)
                
                reqs = self._parse_json_response(response, request=full_prompt)
                if reqs is not None:
                    return reqs
                else:
                    tries-=1
                    print(f"Receiving unparsable response: {response}. Tries left: {tries}")
            except openai.BadRequestError as ex:
                return LlmResponseError(failure_reason="Exception from Llm:", response="",prompt=prompt, caught_exception=ex.message)
        return LlmResponseError(failure_reason="Could not parse LLM response after 3 tries.", response=response, prompt=prompt, caught_exception=None)

