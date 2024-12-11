import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.mlProject.entity.config_entity import DataTransformationConfig
from src.mlProject.utils.common import save_object

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_object(self):
        try:
            numerical_features = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
                                'Visibility (10m)', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)',
                                'Rainfall(mm)', 'Snowfall (cm)']
            
            categorical_features = ['Seasons', 'Holiday', 'Functioning Day']
            
            num_pipeline = StandardScaler()
            cat_pipeline = LabelEncoder()
            
            return num_pipeline, cat_pipeline, numerical_features, categorical_features
            
        except Exception as e:
            raise e
        
    def initiate_data_transformation(self, data_path):
        try:
            df = pd.read_csv(data_path)
            
            num_pipeline, cat_pipeline, numerical_features, categorical_features = self.get_data_transformer_object()
            
            # Transform numerical features
            df[numerical_features] = num_pipeline.fit_transform(df[numerical_features])
            
            # Transform categorical features
            for feature in categorical_features:
                df[f"{feature}_encoded"] = cat_pipeline.fit_transform(df[feature])
            
            # Save preprocessor object
            preprocessor_path = self.config.preprocessed_object_file
            save_object(file_path=preprocessor_path, obj={
                'num_pipeline': num_pipeline,
                'cat_pipeline': cat_pipeline,
                'numerical_features': numerical_features,
                'categorical_features': categorical_features
            })
            
            return df, preprocessor_path
            
        except Exception as e:
            raise e