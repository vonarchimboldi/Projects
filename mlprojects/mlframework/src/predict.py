import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np
from . import dispatcher
from .preprocess import filter_df, encode_df

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")
FEATURE_SELECTION = os.environ.get("FEATURE_SELECTION")

def predict():
    df = pd.read_csv(TEST_DATA)
    predictions = None

    test_idx = df.index.values

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)

        df = encode_df(df)
        #encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
        
        # data is ready to train
        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    return sub
    

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index=False)