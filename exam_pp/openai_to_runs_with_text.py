import os
from typing import Optional
import openai
import requests
import time

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


# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY", None)
if openai.api_key is None:
    raise RuntimeError ("Must set environment variable \"OPENAI_API_KEY\"")



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

# Initialize rate limiter
rate_limiter = OpenAIRateLimiter()

def query_gpt_batch_with_rate_limiting(prompts, rate_limiter):
    responses = []

    for prompt in prompts:
        rate_limiter.reset_if_needed()

        response = openai.Completion.create(engine="gpt-4", prompt=prompt, max_tokens=512)
        responses.append(response.choices[0].text.strip())

        # Update rate limits
        used_tokens = response['usage']['total_tokens']
        rate_limiter.update_limits(used_tokens)

    return responses



class FetchGptResponsesForTrecCar():
    def __init__() 

    def section_prompt(query_title:str, query_heading:str)->str:
        return f"Generate a Wikipedia section on \"{query_heading}\" for an article on \"{query_title}\"."

    def page_prompt(query_title:str)->str:
        return f"Generate a 1000-word long Wikipedia article on \"{query_title}\"."

