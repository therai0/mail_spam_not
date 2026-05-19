
import sys
import re 
import numpy as np 
import pandas as pd 
from bs4 import BeautifulSoup
from pandas import DataFrame 
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from src.exception.exception import CustomeException
from src.entity.artifact_entity import DataIngestionArtifacts,DataTransformationArtifacts
from src.entity.config_entity import DataTransformationConfig
from src.utils.utils import save_numpy_array_data,save_vector_tfidf_model,save_vector_model


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
            for x in data["origin"]:
                # 1. Strip HTML tags
                x = BeautifulSoup(x, "html.parser").get_text()

                # 2. Replace URLs and emails with placeholder tokens
                x = re.sub(r'http\S+|www\S+', ' urltoken ', x)
                x = re.sub(r'\S+@\S+', ' emailtoken ', x)

                # 3. Lowercase early
                x = x.lower()

                # 4. Remove non-alphanumeric (now safe to do)
                x = re.sub(r'[^a-z0-9\s]', '', x)

                # 5. Tokenize
                token = word_tokenize(x)

                # 6. Filter stopwords + lemmatize
                clean_token = [
                    lemmatizer.lemmatize(w)
                    for w in token
                    if w not in stp_words and w.strip() != ""
                ]

                tokens.append(clean_token)
            return tokens
        except Exception as e:
            raise CustomeException(e, sys)
        

    def get_word2vec_model(self,tokens):
        """
        --> Create a word2vec model and return 
        """
        return Word2Vec(tokens,min_count=1)

 
    def text_to_vector_tfidf(self, tokens):
        """
        --> Convert tokens into TF-IDF vectors and return numpy array
        """
        
        try:
            # Convert token list into sentence
            corpus = []

            for sentence in tokens:
                corpus.append(" ".join(sentence))

            # TF-IDF Vectorizer
            tfidf = TfidfVectorizer(
                max_features=3000
            )

            # Fit and transform
            vector = tfidf.fit_transform(corpus)

            # Save TF-IDF model
            save_vector_tfidf_model(
                self.data_transformation_config.text_to_vector_model_path,
                tfidf
            )

            return vector.toarray()
        except Exception as e:
                raise CustomeException(e,sys)

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


    def text_to_vector_bow(self, tokens):
        """
        --> Convert tokens into Bag of Words vectors
        """
        try:

            # Convert token list into sentence
            corpus = []

            for sentence in tokens:
                corpus.append(" ".join(sentence))

            # Bag of Words Vectorizer
            bow = CountVectorizer(
                max_features=3000
            )

            # Fit and transform
            vector = bow.fit_transform(corpus)

            # Save BOW model
            save_vector_tfidf_model(
                self.data_transformation_config.text_to_vector_model_path,
                bow
            )

            return vector.toarray()

        except Exception as e:
            raise CustomeException(e, sys)

    def text_to_vector_ohe(self, tokens):
        """
        --> Convert tokens into One Hot Encoding vectors
        """
        try:
            # Convert token list into sentence
            corpus = []

            for sentence in tokens:
                corpus.append(" ".join(sentence))

            # One Hot Encoding Vectorizer
            ohe = CountVectorizer(
                binary=True,
                max_features=3000
            )

            # Fit and transform
            vector = ohe.fit_transform(corpus)

            # Save OHE model
            save_vector_tfidf_model(
                self.data_transformation_config.text_to_vector_model_path,
                ohe
            )

            return vector.toarray()

        except Exception as e:
            raise CustomeException(e, sys)


    def init_data_transformation(self)->DataTransformationArtifacts:
        try:
            train_data = pd.read_csv(self.data_ingestion_artifacts.train_file_path)
            test_data = pd.read_csv(self.data_ingestion_artifacts.test_file_path)

            y_train = train_data[self.data_transformation_config.target_column]
            y_test = test_data[self.data_transformation_config.target_column]

            X_train_token = self.text_preprocessing(train_data)
            X_test_token = self.text_preprocessing(test_data)

            # word2vec
            # X_train_vec = self.text_to_vector(X_train_token)
            # X_test_vec = self.text_to_vector(X_test_token)

            # tfidf
            X_train_vec = self.text_to_vector_tfidf(X_train_token)
            X_test_vec = self.text_to_vector_tfidf(X_test_token)

            # bow
            # X_train_vec = self.text_to_vector_bow(X_train_token)
            # X_test_vec = self.text_to_vector_bow(X_test_token)

            # ohe
            # X_train_vec = self.text_to_vector_ohe(X_train_token)
            # X_test_vec = self.text_to_vector_ohe(X_test_token)

            train_data = np.c_[X_train_vec,np.array(y_train)]
            test_data = np.c_[X_test_vec,np.array(y_test)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_array_path,train_data)
            save_numpy_array_data(self.data_transformation_config.transformed_test_array_path,test_data)

            return DataTransformationArtifacts(
                train_arr_file_path=self.data_transformation_config.transformed_train_array_path,
                test_arr_file_path=self.data_transformation_config.transformed_test_array_path
            )
        except Exception as e:
            raise CustomeException(e,sys)
