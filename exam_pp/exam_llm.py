
from dataclasses import dataclass
from typing import Optional, List, Dict
from pydantic.v1 import BaseModel

# @dataclass
class LlmResponseError(BaseModel):
    response:str
    prompt:str
    failure_reason:str
    caught_exception: Optional[str]


DIRECT_GRADING_PROMPT_TYPE="direct_grading"