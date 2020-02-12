from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.cluster import hierarchy as hc
from sklearn.ensemble import RandomForestClassifier
from ../categorical import CategoricalFeatures

class Feature_Selection:

    def __init__(self, df):
        self.dataframe = df

    def _vif_selector(self, threshold, mixed = False):
        
        cat_cols = [c for c in self.dataframe.drop([target], axis = 1).columns if self.dataframe[c].dtype != 'int64']
        int_cols = [c for c in self.dataframe.drop([target], axis = 1).columns if self.dataframe[c].dtype == 'int64']
        
        r2 = dict()

        for i in int_cols:
            lm = LinearRegression().fit(self.dataframe.drop(i, axis = 1), self.dataframe[i])
            r2[i] = lm.score(self.dataframe.drop(i, axis = 1), self.dataframe[i])
            
        to_drop = len([ x for x in list(r2.values()) if x > threshold])
        
        if mixed = False:
            return sorted(r2.keys(), key = lambda v: v[1], reverse = True)[to_drop:] + cat_cols
        
        return sorted(r2.keys(), key = lambda v: v[1], reverse = True)[to_drop:]

    def _hc_selector(self):
        
        cols = [c for c in df.columns if not self.dataframe[c].dtype in ('int64', 'float64')]
        cat_feats = CategoricalFeatures(self.dataframe, 
                                    categorical_features=cols, 
                                    encoding_type="label",
                                    handle_na=True)


        df_transformed = cat_feats.fit_transform()
        corr = np.round(scipy.stats.spearmanr(df_transformed).correlation, 4)
        cluster = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage='average')
        clusters = cluster.fit_predict(corr)
        imps = dict(zip(df_transformed.drop(['y'], axis = 1).columns, zip(clf.feature_importances_, clusters)))
        imps_df = pd.DataFrame(imps).transpose().reset_index()
        to_drop = list(imps_df.groupby(1)['index'].agg({0: 'min'})[0])
        
        return df.transformed.drop(to_drop, axis = 1)

