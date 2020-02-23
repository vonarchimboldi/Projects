import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

TESTING = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")
PROBLEM_TYPE = os.environ.get("PROBLEM_TYPE")
DATASET = os.environ.get("DATASET")

df_test = pd.read_csv(TESTING)

if 'class' in df_test.columns:
    actuals = df_test['class']
    df_test = df_test.drop(['class'], axis = 1)
else:
    df_test = df_test
    actuals = np.zeros(len(df_test.index))

print(df_test.columns)
model = joblib.load(os.path.join("models", f"{MODEL}.pkl"))
if MODEL == 'catboost':
    model.save_model('catboost', 'cbm')

if PROBLEM_TYPE == 'regression':
    preds = model.predict(df_test)
elif PROBLEM_TYPE == 'classification':
    preds = model.predict_proba(df_test)[:, 1]
elif PROBLEM_TYPE == 'multiclass':
    preds = model.predict_proba(df_test)
    preds = [np.argmax(p) for p in preds]



sub = pd.DataFrame(np.column_stack((preds, actuals)), columns = ['predictions', 'actuals'])
sub.to_csv(f"models/{PROBLEM_TYPE}_{MODEL}_{DATASET}_outofsample.csv", index = False)
