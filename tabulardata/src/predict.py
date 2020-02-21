import os
from xgboost import XGBClassifier, XGBRegressor
from lightgbm as LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

TESTING = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")
PROBLEM_TYPE = os.environ.get("PROBLEM_TYPE")
DATASET = os.environ.get("DATASET")

df_test = pd.read_csv(TESTING)

if 'TARGET' in df_test.columns:
    actuals = df_test['TARGET']
    df_test = df_test.drop(['TARGET'], axis = 1)
else:
    actuals = np.zeros(len(df_test.index))

model = joblib.load(os.path.join("models", f"{MODEL}.pkl"))

if PROBLEM_TYPE == 'regression':
    preds = model.predict(df_test)
elif PROBLEM_TYPE == 'classification':
    preds = model.predict_proba(df_test)[0]

sub = pd.DataFrame(np.columnstack((preds, actuals)), columns = ['predictions', 'actuals'])
sub.to_csv(f'models/{PROBLEM_TYPE}_{MODEL}_{DATASET}_outofsample.csv', index = False)
