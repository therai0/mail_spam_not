import os 
import sys 
import pickle
import numpy as np 
from src.exception.exception import CustomeException


def save_model(path:str,model)->None:
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path,exist_ok=True)
        with open(path,'wb') as file:
            pickle.dump(model,file)
    except Exception as e:
        raise CustomeException(e,sys)

def load_model(path:str)->object:
    try:
        if not os.path.exists(path):
            raise Exception("Path does not exist")
        with open(path,'rb') as file:
            pickle.load(file)
    except Exception as e:
        raise CustomeException(e,sys)


def save_numpy_array_data(file_path:str,array:np.array):
    "Save numpy array to file path"
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            np.save(file,array)
    except Exception as e:
        raise CustomeException(e,sys)


def load_numy_array_data(file_path:str)->np.array:
    try:
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomeException(e,sys)
