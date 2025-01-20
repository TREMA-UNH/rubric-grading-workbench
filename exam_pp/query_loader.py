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

def tsv_query_loader(query_tsv:Path)-> Dict[str,str]:
    queries:Dict[str,str] = dict()
    with open(query_tsv, "rt", encoding="utf-8") as file:
        for line in file.readlines():
            line = line.strip()
            if len(line)>0:
                splits = line.split(sep=None)
                query_id = splits[0]
                query_text = " ".join(splits[1:]) if len(splits)>1 else ""
                queries[query_id] = query_text
        print(f"Loaded {len(queries)} queries.")
        return queries
    raise RuntimeError(f"Could not load any queries from file {file}.")    


def load_queries(query_path:Path) -> Dict[str,str]:
    if query_path.suffix.lower() == ".tsv":
        return tsv_query_loader(query_path)
    elif query_path.suffix.lower() == ".qrel" or  query_path.suffix.lower() == ".qrels":
        return tsv_query_loader(query_path)
    elif query_path.suffix.lower() == ".json":
        return json_query_loader(query_path)
    else:
        raise RuntimeError(f"Can only load queries from tsv or json file, but received {query_path} with suffix {query_path.suffix}")

def direct_grading_prompt(prompt_class:str, query_id:str, query_text:Optional[str], facet_id:Optional[str], facet_text:Optional[str], self_rater_tolerant:bool)->Optional[DirectGradingPrompt]:
    if query_text is None: 
        raise RuntimeError(f"Query_text is None for query_id {query_id}. This is not allowed.")
    
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
        # raise RuntimeError(f"Prompt class {prompt_class} not supported by the direct_grading_prompt loader.")\
        return None


def direct_grading_prompts(queries:Dict[str,str], prompt_class:str, max_queries:Optional[int], self_rater_tolerant:bool)->Dict[str,List[DirectGradingPrompt]]:
    result:Dict[str,List[DirectGradingPrompt]] = defaultdict(list)

    for query_id, query_text in itertools.islice(queries.items(), max_queries):
        print(query_id, query_text)

        prompt = direct_grading_prompt(prompt_class=prompt_class, query_id = query_id, query_text= query_text, facet_id=None, facet_text=None, self_rater_tolerant=self_rater_tolerant)
        if prompt is not None:
            result[query_id].append(prompt)
        else:
            print(f"Query Loader Warning: {prompt_class} is not a direct grading prompt.")

    return result


