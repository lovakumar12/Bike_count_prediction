import os
import pandas as pd
from src.mlProject.entity.config_entity import DataIngestionConfig
from src.mlProject.utils.common import get_size
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        df = pd.read_csv(self.config.source_data)
        os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
        df.to_csv(self.config.local_data_file, index=False)
        return self.config.local_data_file