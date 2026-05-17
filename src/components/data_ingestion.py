import sys
import os 
from pandas import DataFrame 
import pandas as pd 
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts


from src.exception.exception import CustomeException


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config;
        except Exception as e:
            raise CustomeException(e,sys)
    

    def read_data(self,path)->DataFrame:
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise CustomeException(e,sys)


    def init_data_ingestion(self,path:str):
        try:
            df = self.read_data(path)
            train_df,test_df = train_test_split(df,test_size=self.data_ingestion_config.train_test_ratio)

            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)            
            os.makedirs(dir_path,exist_ok=True)

            train_df.to_csv(self.data_ingestion_config.train_file_path)
            test_df.to_csv(self.data_ingestion_config.test_file_path)

            return DataIngestionArtifacts(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
        except Exception as e:
            raise CustomeException(e,sys)