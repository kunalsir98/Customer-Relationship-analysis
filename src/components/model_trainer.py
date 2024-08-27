import os 
import sys 
from src.exception import CustomException
from src.logger import logging
import numpy as np 
import pandas as pd 
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting data into train and test sets')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Logistic_Regression': LogisticRegression()
            }

            # Evaluate models using a custom evaluation function
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # Get the best model score from the report dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
          
        except Exception as e:
            logging.error('Exception occurred during model training')
            raise CustomException(e, sys)

