import os
import pandas as pd
from sklearn import model_selection

FEATURE_SELECTION = os.environ.get("FEATURE_SELECTION")
CALIBRATION = os.environ.get("CALIBRATION")
TRAINING_DATA = os.environ.get("TRAINING_DATA")

if __name__ == '__main__':
    df = pd.read_csv(TRAINING_DATA)
    print(CALIBRATION)
    #print(FEATURE_SELECTION)
    if CALIBRATION == 'YES':
        df, df_cal = model_selection.train_test_split(df, shuffle = True, test_size = 0.3)
        df_cal.to_csv('input/calibration_set.csv', index = False)
    
    df['kfold'] = -1
    df = df.sample(frac = 1).reset_index(drop = True)

    df, df_valid = model_selection.train_test_split(df, shuffle = True, test_size = 0.3)
    df_valid.to_csv('input/validation_set.csv', index = False)

    #kf = model_selection.StratifiedKFold(n_splits = 2, shuffle = False, random_state = True)

    #for fold, (train_idx, val_idx) in enumerate(kf.split(X = df, y = df.target.values)):
     #   print(len(train_idx), len(val_idx))
      #  df.loc[val_idx, 'kfold'] = fold

    df.to_csv('input/train_folds.csv', index = False)
    