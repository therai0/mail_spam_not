import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam


from src.exception.exception import CustomeException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifacts
from src.utils.utils import load_numy_array_data
from src.utils.utils import save_ml_model


class ModelTraning:
    def __init__(self,model_trainer_config:ModelTrainerConfig):
        try:
            self.model_trainer_config = model_trainer_config
            # self.data_transformer_artifacts = data_transformer_artifacts
        except Exception as e:
            raise CustomeException(e,sys)
    
    def find_best_model_and_train(self,X_train,y_train,X_test,y_test):
        """
        --> Find the best model
        --> train and return 
        """
        try:
            # models = {
            #     "LogisticRegression":LogisticRegression(verbose=1,class_weight={0:1,1:3}),
            #     "SVC":SVC(verbose=1,class_weight={0:1,1:3}),
            #     "GaussianNB":GaussianNB(),
            #     "DecisionTreeClassifier":DecisionTreeClassifier(class_weight={0:1,1:3}),
            #     "RandomForestClassifier":RandomForestClassifier(),
            #     "AdaBoostClassifier":AdaBoostClassifier(),
            #     "GradientBoostingClassifier":GradientBoostingClassifier()
            # }
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
                print(f"{name}:{score}")
                # Finding best model
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
            print(f"{best_model_name}:{best_score}")
            return best_model
        except Exception as e:
            raise CustomeException(e,sys)


    def RNN(self,X_train,y_train,X_test,y_test):
        try:
            X_train_rnn = np.array(X_train)
            X_test_rnn = np.array(X_test)

            # Reshape for RNN
            # (samples, timesteps, features)

            X_train_rnn = X_train_rnn.reshape(
                (X_train_rnn.shape[0], X_train_rnn.shape[1], 1)
            )

            X_test_rnn = X_test_rnn.reshape(
                (X_test_rnn.shape[0], X_test_rnn.shape[1], 1)
            )

            # Build RNN Model

            rnn_model = Sequential()

            rnn_model.add(
                SimpleRNN(
                    units=64,
                    activation='tanh',
                    input_shape=(X_train_rnn.shape[1], 1)
                )
            )

            rnn_model.add(Dropout(0.5))

            rnn_model.add(Dense(1, activation='sigmoid'))

            # Compile model
            rnn_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # Train model
            rnn_model.fit(
                X_train_rnn,
                y_train,
                epochs=5,
                batch_size=32,
                validation_split=0.2
            )

            # Prediction
            y_pred_prob = rnn_model.predict(X_test_rnn)

            y_pred_rnn = (y_pred_prob > 0.5).astype(int)

            rnn_score = accuracy_score(y_test, y_pred_rnn)

            print(f"RNN: {rnn_score}")

            return rnn_model
        except Exception as e:
            raise(e,sys)

       
    def init_model_training(self):
        try:
            train_data = load_numy_array_data("./Artifacts/05_19_2026_16_53_21/data_transformation/transformed/transformed_train.npy")
            test_data = load_numy_array_data("./Artifacts/05_19_2026_16_53_21/data_transformation/transformed/transformed_test.npy")

            X_train = train_data[:,:-1]
            y_train = train_data[:,-1]

            # print(X_train[0])
            # print(len(X_train[0]))
            
            X_test = test_data[:,:-1]
            y_test = test_data[:,-1]
            
            best_model = self.find_best_model_and_train(X_train,y_train,X_test,y_test)
            # best_model = self.RNN(X_train,y_train,X_test,y_test)
            
            save_ml_model(self.model_trainer_config.ml_model_file_path,best_model)
        except Exception as e:
            raise CustomeException(e,sys)