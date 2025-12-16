import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import uniform, randint

import mlflow
import mlflow.sklearn 


logger= get_logger(__name__)

class Model_Training:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.random_search_params = RANDOM_SEARCH_PARAMS
        self.LIGHTGBM_PARAMS = LIGHTGBM_PARAMS

    def load_and_split_data(self):
        try:
            logger.info("Loading  data from {self.train_path}")
            train_df= load_data(self.train_path)

            logger.info("Loading  data from {self.test_path}")
            test_df= load_data(self.test_path)

            X_train= train_df.drop(columns=['booking_status'])
            y_train= train_df['booking_status']

            X_test= test_df.drop(columns=['booking_status'])
            y_test= test_df['booking_status']

            logger.info("Data loading and splitting completed for TRaining")

            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error while loading  data: {e}")
            raise CustomException("Data loading  failed", e)
        
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Starting LightGBM model training with RandomizedSearchCV")

            lgbm_model= lgb.LGBMClassifier(random_state=self.random_search_params["random_state"])

            logger.info("Setting up RandomizedSearchCV for hyperparameter tuning")

            random_search= RandomizedSearchCV(
                estimator= lgbm_model,
                param_distributions= self.LIGHTGBM_PARAMS,
                n_iter= self.random_search_params["n_iter"],
                cv= self.random_search_params["cv"],
                n_jobs= self.random_search_params["n_jobs"],
                verbose= self.random_search_params["verbose"],
                random_state= self.random_search_params["random_state"],
                scoring= self.random_search_params["scoring"]

            )
            

            logger.info("starting our Model Training")
            random_search.fit(X_train, y_train)

            logger.info("Hyper parameter tuning completed.")
            best_params= random_search.best_params_
            best_lgbm_model= random_search.best_estimator_

            logger.info(f"Best parameters found: {best_params}")
            return best_lgbm_model
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException("Model training failed", e)
    
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Starting model evaluation")

            y_pred= model.predict(X_test)

            accuracy= accuracy_score(y_test, y_pred)
            precision= precision_score(y_test, y_pred)
            recall= recall_score(y_test, y_pred)
            f1= f1_score(y_test, y_pred)

            logger.info(f"Model evaluation completed. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException("Model evaluation failed", e)
        
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True )
            logger.info(f"Saving model to {self.model_output_path}")
            joblib.dump(model, self.model_output_path)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error while saving model: {e}")
            raise CustomException("Failed to save model", e)    
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training process")
                logger.info("starting our mlflow experiment")
                logger.info("logging the training and testing dataset to mlflow")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")


                X_train, y_train, X_test, y_test= self.load_and_split_data()

                best_lgbm_model= self.train_lgbm(X_train, y_train)

                evaluation_metrics= self.evaluate_model(best_lgbm_model, X_test, y_test)

                logger.info(f"Evaluation Metrics: {evaluation_metrics}")

                self.save_model(best_lgbm_model)
                
                logger.info("logging the model into mlflow")
                mlflow.log_artifact(self.model_output_path)
                
                logger.info("logging params and metrics to MLflow")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(evaluation_metrics)
                logger.info("Model training process completed successfully")

        except CustomException as ce:
            logger.error(f"CustomException : {str(ce)}")
        
        finally:
            logger.info("Model training process finished")

if __name__ == "__main__":
    trainer= Model_Training(
        PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH
    )
    trainer.run()