from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.data_ingestion import DataIngestion
from src.mlProject.components.data_transformation import DataTransformation
from src.mlProject.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def start_training(self):
        config = ConfigurationManager()
        
        # Data Ingestion
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_path = data_ingestion.initiate_data_ingestion()
        
        # Data Transformation
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        df, preprocessor_path = data_transformation.initiate_data_transformation(data_path)
        
        # Model Training
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model, X_test, y_test = model_trainer.train(df)