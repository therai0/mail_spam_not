import sys

from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataIngestionConfig,TraningConfig,DataTransformationConfig,ModelTrainerConfig
from src.components.model_training import ModelTraning


data_file_path = "./data/email_origin.csv"

if __name__ == "__main__":
    # logging.info("Data Ingestion started")
    traning_config = TraningConfig()
    data_ingestion_config = DataIngestionConfig(traning_config)
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifacts = data_ingestion.init_data_ingestion(data_file_path)
    logging.info("Data Ingestion completed")
    
    logging.info("Text cleaing and vectorization started")
    data_transformation_config =DataTransformationConfig(traing_config=traning_config)
    data_transformation = DataTransformation(data_ingestion_artifacts=data_ingestion_artifacts,data_transformation_config=data_transformation_config)
    data_transformation_artifacts = data_transformation.init_data_transformation()
    logging.info("Text cleaing and vectorization completed")

    # logging.info("Model Traning started")
    # model_trainer_config = ModelTrainerConfig()
    # model_training = ModelTraning(model_trainer_config=model_trainer_config)
    # model_training.init_model_training()
    # logging.info("Model traning finished")
