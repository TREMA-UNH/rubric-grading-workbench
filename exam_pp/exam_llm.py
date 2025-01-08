
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


from typing import TypedDict

class Message(TypedDict):
    '''Message for the ChatCompletions API'''
    role: str
    content: str
    # name: str | None  # Optional, only for function messages.


def convert_to_messages(prompt:str, system_message:Optional[str]=None)->List[Message]:

    messages:List[Message] = list()
    if system_message is not None:
        messages.append({"role":"system", "content":system_message})
    messages.append({"role":"user","content":prompt})

    return messages