
import  sys 
from app.text_preprocessing import TextPreprocessingAndVectorizing
from src.exception.exception import CustomeException
from src.utils.utils import load_ml_model
from src.logging.logger import logging

class Prediction:
    def __init__(self,text:str):
        try:
            self.text = text
        except Exception as e:
            raise CustomeException(e,sys)
    
    def init_prediction(self):
        try:
            logging.info("text preprocessing started")
            text_prerocessing_vectorization = TextPreprocessingAndVectorizing(self.text)
            token = text_prerocessing_vectorization.text_preprocessing()
            logging.info("text vectrozing started")
            vector = text_prerocessing_vectorization.text_to_vector(token=token)
            logging.info("initiation of model prediction")
            logging.info(vector)
            model = load_ml_model("final_model/ml_model.pkl")
            predict = model.predict(vector)
            print(predict)
            if predict == 1:
                return "Spam"
            else:
                return "Not Spam"
            return predict
        except Exception as e:
            raise CustomeException(e,sys)