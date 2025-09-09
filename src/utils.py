
from __future__ import annotations
import json, yaml
from pathlib import Path

def read_yaml(path):
    with open(path,'r') as f: return yaml.safe_load(f)
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w') as f: json.dump(obj,f,indent=2)
