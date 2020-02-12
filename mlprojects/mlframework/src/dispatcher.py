from sklearn import ensemble

MODELS = {
    'randomforest': ensemble.RandomForestClassifier(n_estimators= 100, n_jobs = -1, verbose = 2)
    #'extratrees': ensemble.ExtraTreesClassifier(n_estimators = 100, verbose = 2)
}