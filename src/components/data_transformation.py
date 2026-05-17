import sys
from pandas import DataFrame 
from src.exception.exception import CustomeException
from src.entity.artifact_entity import DataIngestionArtifacts
from src.entity.config_entity import DataTransformationConfig
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re 


class DataTransformation:

    def __init__(self,data_ingestion_artifacts:DataIngestionArtifacts,data_transformation_config:DataTransformationConfig):
        try:
            self.init_ingestion_artifacts = data_ingestion_artifacts
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomeException(e,sys)

    def text_preprocessing(self,data:DataFrame):
        try:
            lemitizer = WordNetLemmatizer()
            data = data.drop([self.data_transformation_config.target_column],axis=1)
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
    
    def init_data_transformation(self):
        try:
            pass 
        except Exception as e:
            raise CustomeException(e,sys)