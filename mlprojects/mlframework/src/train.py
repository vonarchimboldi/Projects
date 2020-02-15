import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
from .preprocess import filter_df, encode_df

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
VALIDATION_DATA = os.environ.get("VALIDATION_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")
FEATURE_SELECTION = os.environ.get("FEATURE_SELECTION")
selector = os.environ.get("FEATURE_SELECTION_METHOD")
problem_type = os.environ.get("PROBLEM_TYPE")
LOSS = os.environ.get("LOSS")

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    train_df = pd.read_csv(TRAINING_DATA)
    valid_df = pd.read_csv(VALIDATION_DATA)
    df_test = pd.read_csv(TEST_DATA)
    #train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
    #valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    if FEATURE_SELECTION == 'YES':
        cols = filter_df(train_df, selector)
    else:
        cols = list(train_df.drop(['target', 'duration'], axis = 1).columns)

    train_df = encode_df(train_df)
    valid_df = encode_df(valid_df)

    train_df = train_df[[*cols, 'target']]
    valid_df = valid_df[[*cols, 'target']]

    train_df = train_df.drop(["target"], axis=1)
    valid_df = valid_df.drop(["target"], axis=1)

    valid_df = valid_df[train_df.columns]

    # data is ready to train
    model = dispatcher.MODELS[MODEL]
    model.fit(train_df, ytrain)

    if problem_type == 'regression':
        preds = model.predict(valid_df)
        print(metrics.rmse(yvalid, preds))

    else:
        preds = model.predict_proba(valid_df)[:, 1]
        print(metrics.roc_auc_score(yvalid, preds))

    #joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    if LOSS == 'quantile':
        joblib.dump(model, f"models/{MODEL}_{FOLD}_{QUANTILE}.pkl")
        
    joblib.dump(model, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")