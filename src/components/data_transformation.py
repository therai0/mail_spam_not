import logging
import sys
import re 
import numpy as np 
import pandas as pd 
from pandas import DataFrame 
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


from src.exception.exception import CustomeException
from src.entity.artifact_entity import DataIngestionArtifacts,DataTransformationArtifacts
from src.entity.config_entity import DataTransformationConfig
from src.utils.utils import save_numpy_array_data,save_vector_model


class DataTransformation:

    def __init__(self,data_ingestion_artifacts:DataIngestionArtifacts,data_transformation_config:DataTransformationConfig):
        try:
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomeException(e,sys)

    def text_preprocessing(self,data:DataFrame):
        """
        --> Text cleaning 
        --> Converting into token 
        --> return tokens:array 
        """
        try:
            lemitizer = WordNetLemmatizer()
            input_data = data["text"]
            stp_words = stopwords.words("english")

            tokens = []
            for x in input_data:
                token = word_tokenize(x)
                clean_token = []
                for word in token:
                    w = re.sub(r'[^a-zA-Z0-9]','',word)
                    if w.lower() not in stp_words and w != "":
                        lemitize = lemitizer.lemmatize(w)
                        clean_token.append(lemitize)
                tokens.append(clean_token)
            return tokens 
        except Exception as e:
            raise CustomeException(e,sys)
    

    def get_word2vec_model(self,tokens):
        """
        --> Create a word2vec model and return 
        """
        return Word2Vec(tokens)


    def text_to_vector(self,tokens):
        """
        --> Convert token into vector and return 
        """
        try:
            model = self.get_word2vec_model(tokens)
            save_vector_model(self.data_transformation_config.text_to_vector_model_path,model)
            
            vector = []
            for sentence in tokens:
                sentence_vector = []

                for word in sentence:
                    if word in model.wv:
                        sentence_vector.append(model.wv[word])

                if len(sentence_vector) == 0:
                    vector.append(np.zeros(model.vector_size))
                else:
                    vector.append(np.mean(sentence_vector,axis=0))
            
            return np.array(vector) 
        except Exception as e:
            raise CustomeException(e,sys)

    def init_data_transformation(self)->DataTransformationArtifacts:
        try:
            train_data = pd.read_csv(self.data_ingestion_artifacts.train_file_path)
            test_data = pd.read_csv(self.data_ingestion_artifacts.test_file_path)

            y_train = train_data[self.data_transformation_config.target_column]
            y_test = test_data[self.data_transformation_config.target_column]

            X_train_token = self.text_preprocessing(train_data)
            X_test_token = self.text_preprocessing(test_data)

            X_train_vec = self.text_to_vector(X_train_token)
            X_test_vec = self.text_to_vector(X_test_token)

            train_data = np.c_[X_train_vec,np.array(y_train)]
            test_data = np.c_[X_test_vec,np.array(y_test)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_array_path,train_data)
            save_numpy_array_data(self.data_transformation_config.transformed_train_array_path,test_data)

            return DataTransformationArtifacts(
                train_arr_file_path=self.data_transformation_config.transformed_train_array_path,
                test_arr_file_path=self.data_transformation_config.transformed_test_array_path
            )
        except Exception as e:
            raise CustomeException(e,sys)