from nonconformist.cp import IcpClassifier, IcpRegressor
from nonconformist.nc import NcFactory
import pandas as pd
import numpy as np
import joblib
import os

TRAINING = os.environ.get("TRAINING_DATA")
CALIBRATION = os.environ.get("CALIBRATION_SET")
TEST = os.environ.get("TEST_DATA")

df_train = pd.read_csv(TRAINING)
df_cal = pd.read_csv(CALIBRATION)
df_test = pd.read_csv(TEST)
trainX, trainY = df_train.drop(['TARGET'],  axis = 1), df_train['TARGET']
calX, calY = df_cal.drop(['TARGET'], axis = 1), df_cal['TARGET']

if 'TARGET' in df_test.columns:
    testX, testY = df_test.drop(['TARGET'], axis = 1), df_test['TARGET']
else:
    testX = df_test

if PROBLEM_TYPE == 'classification':
        
    nc = NcFactory.create_nc(model, normalizer_model=KNeighborsRegressor(n_neighbors=11))	# Create a default nonconformity function
    icp = IcpClassifier(nc)

    icp.fit(trainX, trainY)

    # Calibrate the ICP using the calibration set
    icp.calibrate(calX, calY)

    # Produce predictions for the test set, with confidence 95%
    prediction = icp.predict(testX.to_numpy(), significance=0.05)


else:

    nc = NcFactory.create_nc(model, normalizer_model=KNeighborsRegressor(n_neighbors=11))	# Create a default nonconformity function
    icp = IcpRegressor(nc)			# Create an inductive conformal regressor

    # Fit the ICP using the proper training set
    icp.fit(trainX, trainY)

    # Calibrate the ICP using the calibration set
    icp.calibrate(calX, calY)

    # Produce predictions for the test set, with confidence 95%
    prediction = icp.predict(testX.to_numpy(), significance=0.05)

sub = pd.DataFrame(prediction, columns = ['lower_bound', 'upper_bound'])
sub.to_csv(f"models/{PROBLEM_TYPE}_{MODEL}_{DATASET}_intervals.csv", index = False)