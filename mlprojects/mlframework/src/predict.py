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
if FEATURE_SELECTION == 'YES':
    selector = os.environ.get("FEATURE_SELECTION_METHOD")
else:
    selector = 'none'

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
        
        target = df['target']
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, target, predictions)), columns=["id", "target", "prediction"])
    feat_imps = pd.DataFrame(dict(zip(list(cols), clf.feature_importances_ )).items(), columns = ['feature', 'score'])
    #print(clf.feature_importances_)
    #print(cols)
    return sub, feat_imps
    

if __name__ == "__main__":
    submission, feat_imps = predict()
    submission.to_csv(f"models/{MODEL}_{selector}.csv", index=False)
    feat_imps.to_csv(f"models/{MODEL}_{selector}_feat_imps.csv", index=False)