import os,sys,dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path,object):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file:
            dill.dump(object,file)
        
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models):
    try :
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train,y_train)
            
            y_pred = model.predict(x_test)
            
            test_model_score = r2_score(y_test,y_pred)
            report[f'{model}'[:-2]] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)
        