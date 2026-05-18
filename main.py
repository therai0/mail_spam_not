import sys


from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataIngestionConfig,TraningConfig,DataTransformationConfig


data_file_path = "./data/ham_spam.csv"

if __name__ == "__main__":
    logging.info("Data Ingestion started")
    traning_config = TraningConfig()
    data_ingestion_config = DataIngestionConfig(traning_config)
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifacts = data_ingestion.init_data_ingestion(data_file_path)
    logging.info("Data Ingestion completed")
    
    logging.info("Data transformation and cleaning started")
    data_transformation_config =DataTransformationConfig(traing_config=traning_config)
    data_transformation = DataTransformation(data_ingestion_artifacts=data_ingestion_artifacts,data_transformation_config=data_transformation_config)
    data_transformation_artifacts = data_transformation.init_data_transformation()
    logging.info("Data transformation completed")
