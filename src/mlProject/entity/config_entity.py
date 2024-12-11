from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_data: Path
    local_data_file: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    transformed_data_dir: Path
    preprocessed_object_file: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_dir: Path
    model_file: Path
    params: dict