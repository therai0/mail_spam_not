import os 
import pickle
import sys 
import numpy as np 
from gensim.models import Word2Vec
import pickle
from src.exception.exception import CustomeException



def save_vector_model(path:str,model)->None:
    try:
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir,exist_ok=True)
        model.save(path)
    except Exception as e:
        raise CustomeException(e,sys)

def save_vector_tfidf_model(path:str,model)->None:
    try:
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir,exist_ok=True)
        with open(path,'wb') as file:
            pickle.dump(model,file)
        # model.save(path)
    except Exception as e:
        raise CustomeException(e,sys)

def load_vector_model(path:str):
    try:
        return Word2Vec.load(path)
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


def save_ml_model(path,model)->None:
    try:
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir,exist_ok=True)
        with open(path,'wb') as file:
            pickle.dump(model,file)
    except Exception as e:
        raise CustomeException(e,sys)

def load_ml_model(path):
    try:
        if not os.path.exists(path):
            raise Exception("File path doesn't exist")
        with open(path,'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomeException(e,sys)