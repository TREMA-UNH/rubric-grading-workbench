from collections import defaultdict
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

from .test_bank_prompts import DirectGradingPrompt, FagB, FagB_few, Sun, Sun_few, HELM, Thomas


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
    elif prompt_class == "Sun":
        return Sun(query_id=query_id, query_text=query_text,facet_id=facet_id, facet_text=facet_text)
    elif prompt_class == "Sun_few":
        return Sun_few(query_id=query_id, query_text=query_text,facet_id=facet_id, facet_text=facet_text)
    elif prompt_class == "HELM":
        return HELM(query_id=query_id, query_text=query_text,facet_id=facet_id, facet_text=facet_text)
    elif prompt_class == "Thomas":
        return Thomas(query_id=query_id, query_text=query_text,facet_id=facet_id, facet_text=facet_text)
    else:
        raise RuntimeError(f"Prompt class {prompt_class} not supported by the direct_grading_prompt loader.")\


def direct_grading_prompts(queries:Dict[str,str], prompt_class:str, max_queries:Optional[int])->Dict[str,List[DirectGradingPrompt]]:
    result:Dict[str,List[DirectGradingPrompt]] = defaultdict(list)

    for query_id, query_text in itertools.islice(queries.items(), max_queries):
        print(query_id, query_text)

        prompt = direct_grading_prompt(prompt_class=prompt_class, query_id = query_id, query_text= query_text, facet_id=None, facet_text=None)
        result[query_id].append(prompt)

    return result


