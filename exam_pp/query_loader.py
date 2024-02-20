import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

from .test_bank_prompts import DirectGradingPrompt, FagB, FagB_few


def json_query_loader(query_json:Path)-> Dict[str,str]:

    queries:Dict[str,str]
    with open(query_json, "rt", encoding="utf-8") as file:
        queries = json.load(file)
        return queries

    raise RuntimeError(f"Could not load any queries from file {file}.")    


def direct_grading_prompt(prompt_class:str, query_id:str, query_text:str, facet_id:Optional[str], facet_text:Optional[str])->DirectGradingPrompt:
    if prompt_class == "FagB":
        return FagB(query_id=query_id, query_text=query_text,facet_id=facet_id, facet_text=facet_text)
    elif prompt_class == "FagB_few":
        return FagB_few(query_id=query_id, query_text=query_text,facet_id=facet_id, facet_text=facet_text)
    else:
        raise RuntimeError(f"Prompt class {prompt_class} not supported by the direct_grading_prompt loader.")\


def direct_grading_prompts(queries:Dict[str,str], prompt_class:str, max_queries:Optional[int])->List[DirectGradingPrompt]:
    result = list()

    for query_id, query_text in itertools.islice(queries.items(), max_queries):
        print(query_id, query_text)

        prompt = direct_grading_prompt(prompt_class=prompt_class, query_id = query_id, query_text= query_text, facet_id=None, facet_text=None)
        result.append(prompt)

    return result


