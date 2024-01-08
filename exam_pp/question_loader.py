import itertools
import os
from pathlib import Path
from typing import Tuple, List, Any, Dict, Optional
import json


from question_types import QuestionAnswerablePromptWithChoices
from question_types import *

from pydantic import BaseModel
from typing import List, Any, Optional, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib


import gzip
import re
import json
import itertools
from pathlib import Path


def get_md5_hash(input_string: str) -> str:
    # Convert the string to bytes
    input_bytes = input_string.encode('utf-8')

    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Update the hash object with the bytes
    md5_hash.update(input_bytes)

    # Get the hexadecimal digest of the hash
    hex_digest = md5_hash.hexdigest()

    return hex_digest

def strip_enumeration(s: str) -> str:
    # Regex pattern to match enumeration at the start of the string
    # This pattern matches one or more digits followed by a dot and a space
    pattern = r'^\d+\.\s+'
    # Substitute the matched pattern with an empty string
    return re.sub(pattern, '', s)


def load_naghmehs_questions(question_file:Path)-> List[Tuple[str, List[QuestionPrompt]]]:

    result:List[Tuple[str,List[QuestionPrompt]]] = list()

    content = json.load(open(question_file))
    for query_id, data in content.items():
        qpc_list = list()
        for facet_id, facet_data in data.items():
            query_text = f'{facet_data["title"]} / {facet_data["facet"]}'
            for question_text_with_enumeration in facet_data["questions"]:
                question_text = strip_enumeration(question_text_with_enumeration)
                if len(question_text)>1:
                    question_hash = get_md5_hash(question_text)
                    qpc = QuestionAnswerablePromptWithChoices(question_id=f'{query_id}/{facet_id}/{question_hash}'
                                                , question= question_text
                                                , query_id= query_id
                                                , facet_id = facet_id
                                                , query_text=query_text
                                                , unanswerable_expressions={'unanswerable'}
                                                )

                    qpc_list.append(qpc)
        result.append((query_id, qpc_list))
    return result