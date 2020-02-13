import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
from .preprocess import filter_df, encode_df

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")
FEATURE_SELECTION = os.environ.get("FEATURE_SELECTION")
selector = os.environ.get("FEATURE_SELECTION_METHOD")

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    if FEATURE_SELECTION == 'YES':
        cols = filter_df(df, selector)
    else:
        cols = list(df.drop(['target', 'kfold'], axis = 1).columns)

    train_df = encode_df(train_df)
    valid_df = encode_df(valid_df)

    train_df = train_df[[*cols, 'target', 'kfold']]
    valid_df = valid_df[[*cols, 'target', 'kfold']]

    train_df = train_df.drop(["target", "kfold"], axis=1)
    valid_df = valid_df.drop(["target", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))

    #joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")