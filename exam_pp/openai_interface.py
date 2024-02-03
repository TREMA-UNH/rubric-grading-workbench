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

def query_gpt_batch_with_rate_limiting(prompt:str, gpt_model:str, max_tokens:int):
    result = []

    rate_limiter.wait_if_needed()

    messages = [{"role":"user", "content":prompt}]
    completion = retry_with_exponential_backoff(func=client.chat.completions.create)(model=gpt_model,messages=messages, max_tokens=max_tokens) 

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
