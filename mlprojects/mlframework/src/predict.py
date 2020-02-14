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
NUM_FOLDS = os.environ.get("NUM_FOLDS")
alpha = os.environ.get("alpha")
loss = os.environ.get("LOSS")

if FEATURE_SELECTION == 'YES':
    selector = os.environ.get("FEATURE_SELECTION_METHOD")
else:
    selector = 'none'

def predict():
    df = pd.read_csv(TEST_DATA)
    predictions = None

    test_idx = df.index.values

    for FOLD in range(NUM_FOLDS):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)

        df = encode_df(df)
        #encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
        
        # data is ready to train
        model = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        target = df['target']
        df = df[cols]

        if problem_type == 'regression':
            preds = model.predict(df)
        else:
            preds = model.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= NUM_FOLDS

    sub = pd.DataFrame(np.column_stack((test_idx, target, predictions)), columns=["id", "target", "prediction"])
    #feat_imps = pd.DataFrame(dict(zip(list(cols), model.feature_importances_ )).items(), columns = ['feature', 'score'])
    #print(clf.feature_importances_)
    #print(cols)
    return sub
    

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}_{loss}.csv", index=False)
    #feat_imps.to_csv(f"models/{MODEL}_{selector}_feat_imps_adj.csv", index=False)