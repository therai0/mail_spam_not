import os 
import sys 
from datetime import datetime

from src.exception.exception import CustomeException
from src.constant import training_pipeline

class TraningConfig:
    def __init__(self,timestamp=datetime.now()):
        try:
            timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
            self.artifacts_dir = os.path.join(training_pipeline.artifacts_dir,timestamp)
        except Exception as e:
            raise CustomeException(e,sys)

class DataIngestionConfig:
    def __init__(self,traing_config:TraningConfig):
        try:
            self.ingestion_dir = os.path.join(traing_config.artifacts_dir,training_pipeline.DATA_INGESTION_DIR_NAME)
            self.train_file_path = os.path.join(self.ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR,training_pipeline.train_file)
            self.test_file_path = os.path.join(self.ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR,training_pipeline.test_file)
            self.train_test_ratio = training_pipeline.DATA_INGESTION_TRAIN_AND_TEST_RATIO
        except Exception as e:
            raise CustomeException(e,sys)
        

class DataTransformationConfig:
    def __init__(self,traing_config:TraningConfig):
        try:
            self.transformed_dir = os.path.join(traing_config.artifacts_dir,training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
            self.transformed_train_array_path = os.path.join(self.transformed_dir,training_pipeline.DATA_TRANSFORMATION_TRASFORMED_DIR_NAME,training_pipeline.TRANSFORMED_TRAIN_DF)
            self.transformed_test_array_path = os.path.join(self.transformed_dir,training_pipeline.DATA_TRANSFORMATION_TRASFORMED_DIR_NAME,training_pipeline.TRANSFORMED_TEST_DF)
            self.text_to_vector_model_path = os.path.join(training_pipeline.final_model_dir_name,training_pipeline.vector_model)
            self.target_column = training_pipeline.target_column
        except Exception as e:
            raise CustomeException(e,sys)


class ModelTrainerConfig:
    def __init__(self):
        try:
            self.ml_model_file_path = os.path.join(training_pipeline.final_model_dir_name,training_pipeline.ml_model,)
        except Exception as e:
            raise CustomeException(e,sys)
            