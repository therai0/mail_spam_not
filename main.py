import sys


from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig,TraningConfig


data_file_path = "./data/ham_spam.csv"

if __name__ == "__main__":
    logging.info("Data Ingestion started")
    traning_config = TraningConfig()
    data_ingestion_config = DataIngestionConfig(traning_config)
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifacts = data_ingestion.init_data_ingestion(data_file_path)
    logging.info("Data Ingestion completed")
    print(data_ingestion_artifacts.train_file_path)
    print(data_ingestion_artifacts.test_file_path)
