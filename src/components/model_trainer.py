from dataclasses import dataclass
import os,sys

from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class DataModelConfig:
     train_model_path = os.path.join('artifacts','model.pkl')
     
class ModelTrainer():
    def __init__(self):
        self.model_trainer = DataModelConfig() 
    
    def start_model_trainer(self,train_arr,test_arr):
        try :
            logging.info('splitting train and test dataset')
            # print(train_arr)
            x_train,y_train,x_test,y_test = train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
            # print(x_train)
            # print(y_train)
            
            logging.info('Evaluting the models')
            models = {
                'RandomForestRegressor' : RandomForestRegressor(),
                'DecisionTreeRegressor' : DecisionTreeRegressor(),
                'GradientBoostingRegressor' : GradientBoostingRegressor(),
                'LinearRegression' : LinearRegression(),
                'KNeighborsRegressor' :KNeighborsRegressor(),
                'AdaBoostRegressor' : AdaBoostRegressor()
                }
            model_report : dict = evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            logging.info('model Evaluating completed')
            
            best_model_score = max(list(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            if best_model_score <= 0.6:
                raise CustomException('no best model found')
            logging.info('Best model found both on train and test data')

            save_object(self.model_trainer.train_model_path,best_model)
            logging.info('saving the model')
            
            pred = best_model.predict(x_test)
            
            return r2_score(y_test,pred)

            
        except Exception as e:
            raise CustomException(e,sys)