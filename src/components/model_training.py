import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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
        --> Logistic regression give more accuracy
        --> Hypermeter tuning with logistic regression
        --> train and return model
        """
        try:
           
            param_grid = {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"]
            }

            model = LogisticRegression()
            grid_search_model = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring="accuracy",
                verbose=1
            )
            grid_search_model.fit(X_train, y_train)
            print(grid_search_model.best_params_)

            y_pred = grid_search_model.predict(X_test)

            score = accuracy_score(y_test, y_pred)
            print(score)
        
            return grid_search_model
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
            
            best_model = self.find_best_model_and_train(X_train,y_train,X_test,y_test)
            
            
            save_ml_model(self.model_trainer_config.ml_model_file_path,best_model)
        except Exception as e:
            raise CustomeException(e,sys)