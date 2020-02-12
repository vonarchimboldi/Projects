from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import scipy
from scipy.cluster import hierarchy as hc
from sklearn.ensemble import RandomForestClassifier
from categorical import CategoricalFeatures
from utils import returnhclusters, returnvariableimps

class Feature_Selection:
    def __init__(self, df, selector, vif_threshold = 0.5, mixed = False):
        self.dataframe = df
        self.selector = selector
        self.threshold = vif_threshold
        self.mixed = mixed

    def _vif_selector(self):
        cat_cols = [c for c in self.dataframe.drop(['target'], axis = 1).columns if self.dataframe[c].dtype != 'int64']
        int_cols = [c for c in self.dataframe.drop(['target'], axis = 1).columns if self.dataframe[c].dtype == 'int64']
        
        r2 = dict()

        for i in int_cols:
            df = self.dataframe[int_cols]
            lm = LinearRegression().fit(df.drop(i, axis = 1), df[i])
            r2[i] = lm.score(df.drop(i, axis = 1), df[i])
            
        to_drop = len([ x for x in list(r2.values()) if x > self.threshold])
        
        if self.mixed == False:
            return sorted(r2.keys(), key = lambda v: v[1], reverse = True)[to_drop:] + cat_cols
        
        return sorted(r2.keys(), key = lambda v: v[1], reverse = True)[to_drop:]

    def _hc_selector(self):
        cols = [c for c in self.dataframe.columns if not self.dataframe[c].dtype in ('int64', 'float64')]
        cat_feats = CategoricalFeatures(self.dataframe, 
                                    categorical_features=cols, 
                                    encoding_type="label",
                                    handle_na=True)


        df_transformed = cat_feats.fit_transform()

        clusters = returnhclusters(df_transformed)
        feat_imps = returnvariableimps(df_transformed)

        imps = dict(zip(df_transformed.drop(['target'], axis = 1).columns, zip(feat_imps, clusters)))
        imps_df = pd.DataFrame(imps).transpose().reset_index()
        to_drop = list(imps_df.groupby(1)['index'].agg({0: 'min'})[0])
        
        return list(df_transformed.drop(to_drop, axis = 1).columns)

    def get_features(self):
        print(self.selector)
        if self.selector == 'vif':
            return self._vif_selector()
        elif self.selector == 'hclust':
            return self._hc_selector()
        elif self.selector == 'mixed':
            int_cols = self._vif_selector()
            cat_cols = self._hc_selector()
            return list(set(int_cols + cat_cols))
        else:
            raise Exception('Selection Type Not Understood!')

if __name__ == "__main__":
    df = pd.read_csv("../input/bank_train.csv")
    
    feat_selector = Feature_Selection(df, 
                             selector = "vif", 
                             vif_threshold =0.2,
                             mixed =False)

    selected_features = feat_selector.get_features()
    