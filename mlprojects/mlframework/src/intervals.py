import os
from preprocess import encode_df
import forestci
from nonconformist.cp import IcpClassifier, IcpRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from nonconformist.nc import NcFactory
import pandas as pd
import numpy as np

class UncertaintyQuantifier():
    def __init__(self, problem_type, interval_type, model, 
                trainX, trainY = None, calX = None, calY = None, testX, testY = None):
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

        if problem_type == 'classification':
        
            nc = NcFactory.create_nc(self.model, 
                                    normalizer_model=KNeighborsRegressor(n_neighbors=11))	# Create a default nonconformity function
            icp = IcpClassifier(nc)

            icp.fit(trainX, trainY)

            # Calibrate the ICP using the calibration set
            icp.calibrate(calX, calY)

            # Produce predictions for the test set, with confidence 95%
            prediction = icp.predict(testX, significance=0.05)

            # Print the first 5 predictions
            return prediction
        
        else:

            nc = NcFactory.create_nc(self.model, normalizer_model=KNeighborsRegressor(n_neighbors=11))	# Create a default nonconformity function
            icp = IcpRegressor(nc)			# Create an inductive conformal regressor

            # Fit the ICP using the proper training set
            icp.fit(trainX, trainY)

            # Calibrate the ICP using the calibration set
            icp.calibrate(calX, calY)

            # Produce predictions for the test set, with confidence 95%
            prediction = icp.predict(testX, significance=0.05)

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
        elif:
            return self._jacknife_interval()
        else:
            raise Exception('Interval Type not understood!')
    
    if __name__ == "__main__":
        model = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))

        df_train = pd.read_csv("../input/bank_train.csv")
        df_train_target = df_train['target']
        df_train = df_train[cols]

        CALIBRATION = os.environ.get('CALIBRATION')

        if CALIBRATION == 'YES':
            df_cal = pd.read_csv("../input/calibration_set.csv")
            df_cal = encode_df(df_cal)
            df_cal_target = df_cal['target']
            df_cal = df_cal[cols]

        df_test = pd.read_csv("../input_bank_test.csv")
        df_test_target = df_test['target']
        df_test = df_test[cols]
        
        uc_quantifier = UncertaintyQuantifier(model = model,
                                problem_type = "classification", 
                                interval_type = "conformal",
                                trainX = df_train,
                                trainY = df_train_target,
                                calX = df_cal,
                                calY = df_cal_target,
                                testX = df_test,
                                testY = df_test_target)

        interval = uc_quantifier.get_interval()