import sys
from dataclasses import dataclass
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder ,StandardScaler
import os
 
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path =  os.path.join('artifacts','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        # This function is used to data transformation
        try :
            numeric_values = ['reading_score','writing_score','math_score']
            category_values = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())

                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='most_frequent')),
                    ('one hot encoder',OneHotEncoder()),
                ]
            )
            logging.info('Numerical columns standard scaling completed')
            logging.info('Categorical columns standard scaling completed')
            
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numeric_values),
                    ('cat_pipeline',cat_pipeline,category_values),
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_transformation(self,train_path,test_path):
        try :
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            logging.info('reading train and test data completed')
            logging.info('obtaining preprocessing object')
            
            preprocessing_obj =  self.get_data_transformer_obj()

            target_column_name = 'Average'
            numerical_columns = ['writing_score','reading_score','math_score']
            
            input_feature_train = train_data.drop(columns=[target_column_name,'Total_score'])
            target_feature_train = train_data[target_column_name]
            
            input_feature_test = test_data.drop(columns=[target_column_name,'Total_score'])
            target_feature_test = test_data[target_column_name]

            logging.info('applying preprocessing object on training dataFrame and testing dataFrame')
            
            input_feature_train = input_feature_train.iloc[:,2:]
            input_feature_test = input_feature_test.iloc[:,2:]
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test)
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test)]
            
            
            save_object(self.data_transformation_config.preprocessor_obj_path,preprocessing_obj)
            logging.info('saved preprocessing object')

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_path)

        except Exception as e:
            raise CustomException(e,sys)
        
