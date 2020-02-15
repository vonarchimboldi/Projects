import os
from .preprocess import encode_df
import forestci as fci
from nonconformist.cp import IcpClassifier, IcpRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from nonconformist.nc import NcFactory
import pandas as pd
import numpy as np
import joblib

CALIBRATION = os.environ.get('CALIBRATION')
MODEL = "randomforest"
FOLD = 1
TRAINING_DATA = os.environ.get("TRAINING_DATA")
VALIDATION_DATA = os.environ.get("VALIDATION_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
CALIBRATION_SET = os.environ.get("CALIBRATION_SET")
problem_type = os.environ.get("PROBLEM_TYPE")

class UncertaintyQuantifier():
    def __init__(self, problem_type, interval_type, model, 
                trainX, trainY, calX = None, calY = None, testX=None, testY = None):
        self.problem_type = problem_type
        self.interval_type = interval_type
        self.model = model
        self.trainX = trainX
        self.trainY = trainY
        self.calX = calX
        self.calY = calY
        self.testX = testX
        self.testY = testY

    def _bootstrap_interval(self):
        percentile = 95
        lower_bound = []
        upper_bound = []
        for x in range(len(self.testX)):
            preds = []
            for pred in self.model.estimators_:
                preds.append(pred.predict(self.testX[x])[0])
            lower_bound.append(np.percentile(preds, (100 - percentile) / 2. ))
            upper_bound.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
        return lower_bound, upper_bound

    def _quantile_interval(self):

        upper = self.model.predict(self.testX)
        self.model.set_params(alpha=1.0 - alpha)
        lower = self.model.predict(self.testX)
        sub = pd.DataFrame(np.column_stack((test_idx, target, lower, upper)), columns=["id", "target", "lower", "upper"])

        return sub

    def _conformal_interval(self):

        if self.problem_type == 'classification':
        
            nc = NcFactory.create_nc(self.model, 
                                    normalizer_model=KNeighborsRegressor(n_neighbors=11))	# Create a default nonconformity function
            icp = IcpClassifier(nc)

            icp.fit(self.trainX, self.trainY)

            # Calibrate the ICP using the calibration set
            icp.calibrate(self.calX, self.calY)

            # Produce predictions for the test set, with confidence 95%
            prediction = icp.predict(self.testX.to_numpy(), significance=0.05)

            # Print the first 5 predictions
            return prediction
        
        else:

            nc = NcFactory.create_nc(self.model, normalizer_model=KNeighborsRegressor(n_neighbors=11))	# Create a default nonconformity function
            icp = IcpRegressor(nc)			# Create an inductive conformal regressor

            # Fit the ICP using the proper training set
            icp.fit(self.trainX, self.trainY)

            # Calibrate the ICP using the calibration set
            icp.calibrate(self.calX, self.calY)

            # Produce predictions for the test set, with confidence 95%
            prediction = icp.predict(self.testX.to_numpy(), significance=0.05)

            return prediction

    def _jacknife_interval(self):

        V_IJ_unbiased = fci.random_forest_error(self.model, self.trainX, self.testX)

        return V_IJ_unbiased

    def get_interval(self):

        if self.interval_type == 'bootstrap':
            if self.problem_type == 'regression':
                return self._bootstrap_interval()
            else:
                raise Exception('Cannot be used for this problem type!')

        elif self.interval_type == 'quantile':
            if self.problem_type == 'regression':
                return self._quantile_interval()
            else:
                raise Exception('Cannot be used for this problem type!')

        elif self.interval_type == 'conformal':
            return self._conformal_interval()
        elif self.interval_type == 'jacknife':
            return self._jacknife_interval()
        else:
            raise Exception('Interval Type not understood!')
    
if __name__ == "__main__":
    model = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
    cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))

    print(cols)

    df_train = pd.read_csv(TRAINING_DATA)
    df_train = encode_df(df_train)
    df_train_target = df_train['target']
    df_train = df_train[cols]
    #df_train.drop(['age'], axis = 1, inplace = True)

    if CALIBRATION == 'YES':
        df_cal = pd.read_csv(CALIBRATION_SET)
        df_cal = encode_df(df_cal)
        df_cal_target = df_cal['target']
        df_cal = df_cal[cols]
        #df_cal.drop(['age'], axis = 1, inplace = True)

    df_test = pd.read_csv(TEST_DATA)
    df_test = encode_df(df_test)
    df_test_target = df_test['target']
    df_test = df_test[cols]
    #df_test.drop(['age'], axis = 1, inplace = True)
        
    uc_quantifier = UncertaintyQuantifier(model = model,
                                problem_type = "classification", 
                                interval_type = "jacknife",
                                trainX = df_train,
                                trainY = df_train_target,
                                calX = df_cal,
                                calY = df_cal_target,
                                testX = df_test,
                                testY = df_test_target)

    interval = uc_quantifier.get_interval()
    print(interval.shape)