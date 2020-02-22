import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from .hyperparamtuner import TuneParams

TRAINING = os.environ.get("TRAINING_DATA")
VALIDATION = os.environ.get("VALIDATION_DATA")
MODEL = os.environ.get("MODEL")
PROBLEM_TYPE = os.environ.get("PROBLEM_TYPE")
DATASET = os.environ.get("DATASET")

class TrainModel:
    def __init__(self, training_data, validation_data, model, problem_type, params):
        self.training_data = training_data
        self.validation_data = validation_data
        self.model = model
        self.problem_type = problem_type
        self.params = params

    def _xgboost(self):
        
        params = self.params

        if self.problem_type == 'regression':
            model = XGBRegressor(**params)

            model.fit(self.training_data.drop(["TARGET"], axis = 1), self.training_data['TARGET'])

            preds = model.predict(self.validation_data.drop(['TARGET'], axis = 1))

            return model, preds, self.validation_data['TARGET']

        elif self.problem_type == 'classification':
            model = XGBClassifier(**params)

            model.fit(self.training_data.drop(["TARGET"], axis = 1), self.training_data['TARGET'])

            preds = model.predict_proba(self.validation_data.drop(['TARGET'], axis = 1))[0]

            return model, preds, self.validation_data['TARGET']

        else:
            raise Exception("Problem Type not supported")
        

    def _lightgbm(self):
        
        params = self.params
        
        if self.problem_type == 'regression':
            model = lgb.LGBMRegressor(**params)

            model = model.fit(self.training_data.drop(['TARGET'], axis = 1), self.training_data['TARGET'],
                         eval_set = [(self.validation_data.drop(['TARGET'], axis = 1), self.validation_data['TARGET'])],
                         eval_metric = 'l1', early_stopping_rounds = 5)

            preds = model.predict(self.validation_data.drop(['TARGET'], axis = 1), model.best_iteration_)

        elif self.problem_type == 'classification':
            model = lgb.LGBMClassifier(**params)

            model = model.fit(self.training_data.drop(['TARGET'], axis = 1), self.training_data['TARGET'],
                         eval_set = [(self.validation_data.drop(['TARGET'], axis = 1), self.validation_data['TARGET'])],
                         eval_metric = 'logloss', early_stopping_rounds = 5)

            preds = model.predict_proba(self.validation_data.drop(['TARGET'], axis = 1), model.best_iteration_)[0]
        else:
            raise Exception("Problem Type not supported!")

        return model, preds, self.validation_data['TARGET']

    def _catboost(self):

        params = self.params
        cat_features = [i for i, j in enumerate(list(self.training_data.drop(['TARGET'], axis = 1).columns)) if self.training_data[j].dtype != 'int64']
        
        if self.problem_type == 'regression':
            model = CatBoostRegressor()
            print(type(model))
            model.set_params(**params)

            model.fit(self.training_data.drop(["TARGET"], axis = 1), self.training_data['TARGET'],
                     cat_features = cat_features, 
                     eval_set = (self.validation_data.drop(['TARGET'], axis = 1), self.validation_data['TARGET']))

            preds = model.predict(self.validation_data.drop(['TARGET'], axis = 1))

            return model, preds, self.validation_data['TARGET']

        elif self.problem_type == 'classification':
            model = CatBoostClassifier()
            model.set_params(**params)
            model.fit(self.training_data.drop(["TARGET"], axis = 1), self.training_data['TARGET'],
                     cat_features = cat_features, 
                     eval_set = (self.validation_data.drop(['TARGET'], axis = 1), self.validation_data['TARGET']))

            preds = model.predict_proba(self.validation_data.drop(['TARGET'], axis = 1))[0]

            return model, preds, self.validation_data['TARGET']

        else:
            raise Exception("Problem Type not supported")
        

    def train_and_validate(self):

        if self.model == "xgboost":
            return self._xgboost()
        elif self.model == "lightgbm":
            return self._lightgbm()
        elif self.model == "catboost":
            return self._catboost()
        else:
            raise Exception("Model not supported")

if __name__ == "__main__":
    df_train = pd.read_csv(TRAINING)
    df_valid = pd.read_csv(VALIDATION)
    tuner = TuneParams(pd.concat([df_train, df_valid], axis = 0), MODEL, PROBLEM_TYPE)
    params = tuner.get_params()

    Learner = TrainModel(df_train, df_valid, MODEL, PROBLEM_TYPE, params)
    model, preds, actuals = Learner.train_and_validate()

    sub = pd.DataFrame(np.column_stack((preds, actuals)), columns=["prediction", "target"])
    sub.to_csv(f"models/{PROBLEM_TYPE}_{MODEL}_{DATASET}_insample.csv", index = False)
    joblib.dump(model, f"models/{MODEL}.pkl")



