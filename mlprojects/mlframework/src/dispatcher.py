from sklearn import ensemble

MODELS = {
    'randomforest': ensemble.RandomForestClassifier(n_estimators= 100, min_samples_leaf = 10, max_features = 'sqrt', n_jobs = -1, verbose = 2),
    'gradientboosting': ensemble.GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.05, max_features = 'sqrt'),
    'gbrquantileloss': ensemble.GradientBoostingRegressor(loss = 'quantile', n_estimators = 100, learning_rate = 0.1, max_features = 'sqrt'),
    'gbclassifier': ensemble.GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, max_features = 'sqrt')
    }