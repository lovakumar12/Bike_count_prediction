import numpy as np
import pandas as pd
from src.mlProject.utils.common import load_object
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = load_object(Path('artifacts/model_trainer/model.pkl'))
        self.preprocessor = load_object(Path('artifacts/data_transformation/preprocessor.pkl'))
        
    def predict(self, features):
        try:
            # Create DataFrame from features
            df = pd.DataFrame([features])
            
            # Transform numerical features
            num_features = self.preprocessor['numerical_features']
            df[num_features] = self.preprocessor['num_pipeline'].transform(df[num_features])
            
            # Transform categorical features
            cat_features = self.preprocessor['categorical_features']
            for feature in cat_features:
                df[f"{feature}_encoded"] = self.preprocessor['cat_pipeline'].transform(df[feature])
            
            # Select features for prediction
            features_for_prediction = num_features + [f"{feat}_encoded" for feat in cat_features]
            
            # Make prediction
            prediction = self.model.predict(df[features_for_prediction])
            
            return prediction
            
        except Exception as e:
            raise e