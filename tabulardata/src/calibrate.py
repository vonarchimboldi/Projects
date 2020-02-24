from sklearn.calibration import CalibratedClassifierCV
import os
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
import pandas as pd
import numpy as np
import joblib

TRAINING = os.environ.get("TRAINING_DATA")
VALIDATION = os.environ.get("VALIDATION_DATA")
MODEL = os.environ.get("MODEL")
METHOD = os.environ.get("CALIBRATIONMETHOD")

class CalibratedPredictions:
    def __init__(self, training_data, test_data, model, method):

        self.training_data = training_data
        self.test_data = test_data
        self.model = model
        self.method = method

    def _binning(self):
        X_train, y_train, X_test, y_test = self.get_data()
        preds = model.predict_proba(X_test)[:, 1]
        predictions = pd.DataFrame(preds, columns = ['predictions'])
        predictions['bin'] = pd.cut(preds, bins = 10)
        predictions['predictions'] = predictions.groupby('bin')['predictions'].transform(lambda x: x.mean())

        return predictions['predictions'], y_test

    def _sigmoid(self):
        X_train, y_train, X_test, y_test = self.get_data()
        model_sigmoid = CalibratedClassifierCV(model, cv=2, method='sigmoid')
        model_sigmoid.fit(X_train.values, y_train.values)
        prob_pos_sigmoid = model_sigmoid.predict_proba(X_test.values)[:, 1]

        return prob_pos_sigmoid, y_test

    def _isotonic(self):
        X_train, y_train, X_test, y_test = self.get_data()
        model_isotonic = CalibratedClassifierCV(model, cv=2, method='isotonic')
        model_isotonic.fit(X_train.values, y_train.values)
        prob_pos_isotonic = model_isotonic.predict_proba(X_test.values)[:, 1]

        return prob_pos_isotonic, y_test

    def get_data(self):
        X_train, y_train = self.training_data.drop(['TARGET'], axis = 1), self.training_data['TARGET']
        X_test, y_test = self.test_data.drop(['TARGET'], axis = 1), self.test_data['TARGET']

        return X_train, y_train, X_test, y_test

    def get_predictions(self):

        if self.method == 'binning':
            return self._binning()
        elif self.method == 'sigmoid':
            return self._sigmoid()
        elif self.method == 'isotonic':
            return self._isotonic()
        else:
            raise Exception("Method not supported!")

if __name__ == '__main__':
    model = joblib.load(os.path.join("models", f"{MODEL}.pkl"))
    df_train = pd.read_csv(TRAINING)
    df_validation = pd.read_csv(VALIDATION)
    Calibrator = CalibratedPredictions(df_train, df_validation, model, method = METHOD)
    preds, actuals = Calibrator.get_predictions()

    sub = pd.DataFrame(np.column_stack((preds, actuals)), columns = ['prediction', 'actual'])
    sub.to_csv(f"models/{MODEL}__{METHOD}_calibratedpreds.csv", index = False)