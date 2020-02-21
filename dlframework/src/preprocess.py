import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
import pandas as pd
import numpy as np
from .encode import encode_df
from sklearn import preprocessing


class PreprocessData:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        int_cols = [i for i in list(self.data.columns) if self.data[i].dtype == 'int64']
        self.intcols = int_cols

        cat_cols = []
        cat_cols = [i for i in list(self.data.columns) if self.data[i].dtype != 'int64']
        self.catcols = cat_cols
    
    
    def _lightgbm(self):
        
        return self.data

    def _catboost(self):

        return self.data
    
    def _xgb(self):
        
        
        if len(self.catcols) > 0:
            return encode_df(self.data, self.catcols)
        else:
            return self.data

    def get_data(self):

        if self.model == 'lightgbm':
            return self._lightgbm()
        elif self.model == 'catboost':
            return self._catboost()
        elif self.model == 'xgboost':
            return self._xgb()
        else:
            raise Exception('Model Type not Supported!')


if __name__ == '__main__':
    df = pd.read_csv("../data/raw/poker_hands/train.csv", index_col = 0)
    preprocessor = PreprocessData(model = "xgboost",
                                  data = df)
    df  = preprocessor.get_data()
    