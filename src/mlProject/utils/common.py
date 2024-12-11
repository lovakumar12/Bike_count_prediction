import os
import yaml
import json
import joblib
from box import ConfigBox
from pathlib import Path
from typing import Any
import pandas as pd
from ensure import ensure_annotations

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns ConfigBox"""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories"""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"created directory at: {path}")

@ensure_annotations
def save_object(file_path: Path, obj: Any):
    """save joblib object"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise e

@ensure_annotations
def load_object(file_path: Path) -> Any:
    """load joblib object"""
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise e

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data"""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data"""
    with open(path) as f:
        content = json.load(f)
    return ConfigBox(content)

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB"""
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"