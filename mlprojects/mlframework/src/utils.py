import numpy as np
import scipy
from scipy.cluster import hierarchy as hc
from sklearn.ensemble import RandomForestClassifier

def returnhclusters(df):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    cluster = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage='average')
    clusters = cluster.fit_predict(corr)
    return clusters

def returnvariableimps(df, target):
    clf = RandomForestClassifier(n_estimators = 50, n_jobs = -1, verbose = 1)
    clf.fit(df.drop(['target'], axis = 1), df['target'])

    return clf.feature_importances_