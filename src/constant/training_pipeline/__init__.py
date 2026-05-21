import numpy as np 
import os 


"""
Defining some common constant var
"""
train_file:str = "train.csv"
test_file:str = "test.csv"
target_column:str = "label"
artifacts_dir = "Artifacts"
vector_model = "vector.pkl"
final_model_dir_name = "final_model"
ml_model = "ml_model.pkl"

"""
Constant variable for data ingestion
"""
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_AND_TEST_RATIO = 0.2


"""
Constant varaible for data transformation
"""
DATA_TRANSFORMATION_DIR_NAME = "data_transformation"
DATA_TRANSFORMATION_TRASFORMED_DIR_NAME = "transformed"
TRANSFORMED_TRAIN_DF = "transformed_train.npy"
TRANSFORMED_TEST_DF = "transformed_test.npy"
