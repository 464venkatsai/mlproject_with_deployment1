import sys ,os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig :
    train_data_path : str=os.path.join('artifacts','train.csv')
    test_data_path : str=os.path.join('artifacts','test.csv')
    raw_data_path : str=os.path.join('artifacts','raw.csv')
    
class DataIngestion():
    def __init__(self):
        self.DataIngestionConfig = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Entered logging')
        try :
            data = pd.read_csv('notebook\dataset\stud.csv')
            
            logging.info('reading the dataset as dataframe')
            os.makedirs(os.path.dirname(self.DataIngestionConfig.train_data_path),exist_ok=True)
            data.to_csv(self.DataIngestionConfig.raw_data_path,index=False,header=True)
            logging.info('train test split started')

            train_set , test_set = train_test_split(data,test_size=0.2,random_state=45)
            train_set.to_csv(self.DataIngestionConfig.train_data_path)
            test_set.to_csv(self.DataIngestionConfig.test_data_path)
            
            logging.info('Ingestion completed')
            
            return self.DataIngestionConfig.train_data_path , self.DataIngestionConfig.test_data_path
        
        except Exception as e:
            raise CustomException(e,sys)
    
if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr , test_arr , _ = data_transformation.start_data_transformation(train_data_path,test_data_path)
    
    model_trainer = ModelTrainer()
    r2_score = model_trainer.start_model_trainer(train_arr,test_arr)
    print('Model Accuracy : ',r2_score*100)
    