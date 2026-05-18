from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    train_file_path:str
    test_file_path:str 

@dataclass 
class DataTransformationArtifacts:
    train_arr_file_path:str
    test_arr_file_path:str