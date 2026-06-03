
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


import sys
import re 
import numpy as np 
import pandas as pd 
from bs4 import BeautifulSoup
from pandas import DataFrame 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer



from src.exception.exception import CustomeException
from src.entity.artifact_entity import DataIngestionArtifacts,DataTransformationArtifacts
from src.entity.config_entity import DataTransformationConfig
from src.utils.utils import save_numpy_array_data,save_vector_tfidf_model


class DataTransformation:

    def __init__(self,data_ingestion_artifacts:DataIngestionArtifacts,data_transformation_config:DataTransformationConfig):
        try:
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomeException(e,sys)

    def text_preprocessing(self, data: DataFrame):
        try:
            lemmatizer = WordNetLemmatizer()
            stp_words = set(stopwords.words("english"))  # set = O(1) lookup

            tokens = []
            for x in data[self.data_transformation_config.text_column]:
               
                x = BeautifulSoup(x, "html.parser").get_text()

                x = re.sub(r'http\S+|www\S+', ' urltoken ', x)
                x = re.sub(r'\S+@\S+', ' emailtoken ', x)

                x = x.lower()

                x = re.sub(r'[^a-z0-9\s]', '', x)

                token = word_tokenize(x)

                clean_token = [
                    lemmatizer.lemmatize(w)
                    for w in token
                    if w not in stp_words and w.strip() != ""
                ]

                tokens.append(clean_token)
            return tokens
        except Exception as e:
            raise CustomeException(e, sys)
        

    def text_to_vector_tfidf(self, X_train_token,X_test_token):
        """
        --> Convert tokens into TF-IDF vectors and return numpy array
        """
        try:
            # Convert token list into sentence
            X_train_corpus = []
            X_test_corpus = []

            for sentence in X_train_token:
                X_train_corpus.append(" ".join(sentence))
            
            for sentence in X_test_token:
                X_test_corpus.append(" ".join(sentence))

            # TF-IDF Vectorizer
            tfidf = TfidfVectorizer(
                max_features=3000
            )

            # Fit and transform
            train_vector = tfidf.fit_transform(X_train_corpus)
            test_vector = tfidf.transform(X_test_corpus)

            save_vector_tfidf_model(self.data_transformation_config.text_to_vector_model_path,tfidf)

            return (
                train_vector.toarray(),
                test_vector.toarray()
            )
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

            # tfidf
            X_train_vec,X_test_vec = self.text_to_vector_tfidf(X_train_token,X_test_token)

            
            train_data = np.c_[X_train_vec, np.asarray(y_train, dtype=np.float64)]
            test_data = np.c_[X_test_vec, np.asarray(y_test, dtype=np.float64)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_array_path,train_data)
            save_numpy_array_data(self.data_transformation_config.transformed_test_array_path,test_data)

            return DataTransformationArtifacts(
                train_arr_file_path=self.data_transformation_config.transformed_train_array_path,
                test_arr_file_path=self.data_transformation_config.transformed_test_array_path
            )
        except Exception as e:
            raise CustomeException(e,sys)
