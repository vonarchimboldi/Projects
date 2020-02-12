import os
import pandas as pd
from sklearn import model_selection

FEATURE_SELECTION = os.environ.get("SELECTION")

if __name__ == '__main__':
    df = pd.read_csv('input/bank-train.csv')
    df['kfold'] = -1

    df = df.sample(frac = 1).reset_index(drop = True)

    kf = model_selection.StratifiedKFold(n_splits = 5, shuffle = False, random_state = True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X = df, y = df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    df.to_csv('input/train_folds.csv', index = False)