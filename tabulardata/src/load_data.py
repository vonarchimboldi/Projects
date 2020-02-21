import os
import pandas as pd
from src.preprocess import PreprocessData
from sklearn import model_selection


TRAIN = os.environ.get("TRAINING_DATA")
TEST = os.environ.get("TEST_DATA")
df_train = pd.read_csv(TRAIN)
df_test = pd.read_csv(TEST)
preprocessor = PreprocessData(model = "xgboost",
                                  data = df_train)
df  = preprocessor.get_data()


df, df_cal = model_selection.train_test_split(df, shuffle = True, test_size = 0.3)#, stratify = df['TARGET'])
df_cal.to_csv('data/primary/sarcos_inv/calibration_set.csv', index = False)
    
df = df.sample(frac = 1).reset_index(drop = True)

df, df_valid = model_selection.train_test_split(df, shuffle = True, test_size = 0.2)#, stratify = df['TARGET'])
df_valid.to_csv('data/primary/sarcos_inv/validation_set.csv', index = False)

df.to_csv('data/primary/sarcos_inv/test.csv', index = False)