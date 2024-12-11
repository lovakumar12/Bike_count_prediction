from sklearn.ensemble import RandomForestRegressor
from src.mlProject.entity.config_entity import ModelTrainerConfig
from src.mlProject.utils.common import save_object
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self, df):
        try:
            features = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
                       'Visibility (10m)', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)',
                       'Rainfall(mm)', 'Snowfall (cm)', 'Seasons_encoded', 'Holiday_encoded',
                       'Functioning Day_encoded']
            
            target = 'Rented Bike Count'
            
            X = df[features]
            y = df[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(**self.config.params)
            model.fit(X_train, y_train)
            
            save_object(file_path=self.config.model_file, obj=model)
            
            return model, X_test, y_test
            
        except Exception as e:
            raise e