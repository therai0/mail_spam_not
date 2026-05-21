
import sys 
import nltk
import re

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.exception.exception import CustomeException
from src.utils.utils import load_vector_model


class TextPreprocessingAndVectorizing:
    def __init__(self,text:str):
        try:
            self.text = text
        except Exception as e:
            raise CustomeException(e,sys)


    def text_preprocessing(self):
            try:
                lemmatizer = WordNetLemmatizer()
                stp_words = set(stopwords.words("english")) 
                
                text = self.text 
                text = re.sub(r'http\S+|www\S+', ' urltoken ', text)
                text = re.sub(r'\S+@\S+', ' emailtoken ', text)
                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
                text = text.lower()
                token = word_tokenize(text)
                clean_token = [
                        lemmatizer.lemmatize(w)
                        for w in token
                        if w not in stp_words and w.strip() != ""
                    ]
                return clean_token
            except Exception as e:
                raise CustomeException(e, sys)

    def text_to_vector(self,token):
        try:
            sentence = "".join(token)
            corpus = [sentence]
            vector_model = load_vector_model("final_model/vector.pkl")
            vector = vector_model.transform(corpus)
            return vector.toarray()
        except Exception as e:
            raise CustomeException(e,sys)
