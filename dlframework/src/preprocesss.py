from fastai.tabular import *
import lightgbm as lgb
import catboost as catboost
import xgboost as xbg`

class PreprocessData():
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.int_cols = []
        self.cat_cols = []
        self.data.int_cols.append([i for i in list(self.data.columns) if self.data[i].dtype == 'int64']
        self.data.cat_cols.append([i for in list(self.data.columns) if self.data[i].dtype != 'int64'])
    
    def _fastai(self):
        pass
    
    def _lightgbm(self):
        pass

    def _catboost(self);
        pass
    
    def _xgb(self):
        pass

    def get_data(self):
        if self.model == 'fastai':
            return self._fastai()
        elif self.model == 'lightgbm':
            return self._lightgbm()
        elif self.model == 'catboost':
            return self._catboost()
        elif self.model == 'xgboost':
            return self._xgb()
        else:
            raise Exception('Model Type not Supported!')

