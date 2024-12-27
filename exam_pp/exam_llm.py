
from dataclasses import dataclass
from typing import Optional


@dataclass
class LlmResponseError():
    failure_reason:str
    caught_exception: Optional[Exception]


DIRECT_GRADING_PROMPT_TYPE="direct_grading"