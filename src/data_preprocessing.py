import os
import pandas as pd
import numpy as np

# Correct imports: import the function/class from the modules
from src.logger import get_logger
from src.custom_exception import CustomException

from config.paths_config import *
from utils.common_functions import read_yaml, load_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class Data_Processor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.prcessed_dir = processed_dir
        self.config = read_yaml(config_path)

        if not os.path.exists(self.prcessed_dir):
            os.makedirs(self.prcessed_dir)

    def preprocess_data(self,df):
        try:
            logger.info("Starting data preprocessing")

            logger.info("lets pass dropping unnecessary columns")
            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace= True)
            df.drop_duplicates(inplace=True)
            
            cat_col= self.config["data_processing"]["categorical_columns"]
            num_col= self.config["data_processing"]["numerical_columns"]

            logger.info("Label Encoding")

            label_encode= LabelEncoder()
            mappings= {}
            for col in cat_col:
                df[col]= label_encode.fit_transform(df[col])
                mappings[col]= {label: code for label, code in zip(label_encode.classes_, label_encode.transform(label_encode.classes_))}

            logger.info("Label Mappings")
            for col, mappings in mappings.items():
                logger.info(f"{col}: {mappings}")

            logger.info("Skewness Handling using SMOTE")
            skewness_threshold= self.config["data_processing"]["skewness_threshold"]
            skewness= df[num_col].apply(lambda x: x.skew()).abs()

            for column in skewness[skewness > skewness_threshold].index:
                df[column]= np.log1p(df[column])
            return df
        except Exception as e:
            logger.error(f"Error during data preprocessing {e}")
            raise CustomException("Data preprocessing failed", e)
        
    def balance_data(self, df):
        try:
            X= df.drop(columns='booking_status')
            y= df['booking_status']
            smote= SMOTE()
            X_resampled, y_resampled= smote.fit_resample(X, y)
            balanced_df= pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['booking_status']= y_resampled
            logger.info("Data balancing completed using SMOTE")
            return balanced_df
        except Exception as e:
            logger.error(f"Error during data balancing {e}")
            raise CustomException("data balancing error", e)
        
    def select_features(self, df):
        try:
            logger.info("Starting feature selection step")
            x= df.drop(columns='booking_status')
            y= df['booking_status']
            model= RandomForestClassifier(random_state=42)
            model.fit(x, y)
            feature_importance= model.feature_importances_
            feature_importance_df= pd.DataFrame({
                'feature': x.columns, 
                'importance': feature_importance
                })
            
            top_features_importance_df= feature_importance_df.sort_values(by='importance', ascending=False)


            top_10_features= top_features_importance_df['feature'].head(10).values
            top_10_df= df[top_10_features.tolist()+ ['booking_status']]

            logger.info(f"feature selected: {top_10_features}")

            logger.info(f"Feature selection completed.")
            return top_10_df
        except Exception as e:
            logger.error(f"Error during feature selection {e}")
            raise CustomException("Feature selection failed", e)
        
    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving processed data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info("Data saved successfully to the given file path")
        except Exception as e:
            logger.error(f"Error while saving data to {file_path}: {e}")
            raise CustomException("Failed to save data", e)
        
    def process(self):
        try:
            logger.info("Loading data from raw directory")
            train_df= load_data(self.train_path)
            test_df= load_data(self.test_path)

            train_df= self.preprocess_data(train_df)
            test_df= self.preprocess_data(test_df)

            train_df= self.balance_data(train_df)
            test_df= self.balance_data(test_df)

            train_df= self.select_features(train_df)
            test_df= test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed successfully")
        except Exception as e:
            logger.error(f"Error during data processing pipeline{e}")
            raise CustomException("Data processing failed", e)

if __name__ == "__main__":
    processor= Data_Processor(TRAIN_FILE_PATH,TRAIN_FILE_PATH, PROCESSED_DIR,CONFIG_PATH)
    processor.process()    
        