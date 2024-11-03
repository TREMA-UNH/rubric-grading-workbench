from pydantic.v1 import BaseModel,  __version__ as pydantic_version
import json

# Check Pydantic version
PYDANTIC_V2 = pydantic_version.startswith("2")

if PYDANTIC_V2:
    from pydantic.json import pydantic_encoder

def pydantic_dump(item:BaseModel)->str:
    if PYDANTIC_V2:
        # Use json.dumps with pydantic_encoder for Pydantic version 2
        json_str = json.dumps(item, default=pydantic_encoder, exclude_none=True)
    else:
        # Use .json() method for Pydantic version 1
        json_str = item.json(exclude_none=True)    
    return json_str
