import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from src.exception.exception import CustomeException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifacts
from src.utils.utils import load_numy_array_data
from src.utils.utils import save_ml_model


class ModelTraning:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformer_artifacts:DataTransformationArtifacts):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformer_artifacts = data_transformer_artifacts
        except Exception as e:
            raise CustomeException(e,sys)
    
    def find_best_model_and_train(self,X_train,y_train,X_test,y_test):
        """
        --> Find the best model
        --> train and return 
        """
        try:
            models = {
                "LogisticRegression":LogisticRegression(verbose=1),
                "SVC":SVC(verbose=1),
                "GaussianNB":GaussianNB(),
                "DecisionTreeClassifier":DecisionTreeClassifier(),
                "RandomForestClassifier":RandomForestClassifier(),
                "AdaBoostClassifier":AdaBoostClassifier(),
                "GradientBoostingClassifier":GradientBoostingClassifier()
            }
            best_model = None
            best_score = 0
            best_model_name = ""

            for name, model in models.items():
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                score = accuracy_score(y_test, y_pred)
               
                # Finding best model
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
            print(f"{best_model_name}:{best_score}")
            return best_model
        except Exception as e:
            raise CustomeException(e,sys)

    def init_model_training(self):
        try:
            train_data = load_numy_array_data(self.data_transformer_artifacts.train_arr_file_path)
            test_data = load_numy_array_data(self.data_transformer_artifacts.test_arr_file_path)

            X_train = train_data[:,:-1]
            y_train = train_data[:,-1]

            X_test = test_data[:,:-1]
            y_test = test_data[:,-1]
            
            best_model = self.find_best_model(X_train,y_train,X_test,y_test)
            
            save_ml_model(self.model_trainer_config.ml_model_file_path,best_model)
        except Exception as e:
            raise CustomeException(e,sys)