
from dataclasses import dataclass
from typing import Optional


@dataclass
class LlmResponseError():
    failure_reason:str
    caught_exception: Optional[Exception]

