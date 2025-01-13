import os
import sys
import dill
import numpy as np
import pandas as pd

from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i, (model_name, model) in enumerate(models.items()):
            param = params.get(model_name, {})

            try:
                print(f"Evaluating model: {model_name} with params: {param}")
                
                gs = GridSearchCV(model, param, cv=3)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                report[model_name] = test_model_score

            except Exception as inner_e:
                print(f"Error with model {model_name}: {inner_e}")
                continue

        return report
    except Exception as e:
        raise CustomException(e, sys)
