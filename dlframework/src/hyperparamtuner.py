from xgboost import XGBClassifier, XGBRegressor
from lightgbm as LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

class TuneParams:
    def __init__(self, training_data, validation_data, model, problem_type, params):
        self.training_data = training_data
        self.validation_data = validation_data
        self.model = model
        self.problem_type = problem_type
        self.params = params

    def _xgboost(self):
        
        param_dist = {'n_estimators': stats.randint(150, 500),
              'learning_rate': stats.uniform(0.01, 0.07),
              'subsample': stats.uniform(0.3, 0.7),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.45),
              'min_child_weight': [1, 2, 3]
             }

        if self.problem_type == 'regression':
            model_xgb = XGBRegressor()
            model = RandomizedSearchCV(model_xbg,
                                 param_distributions = param_dist, 
                                 n_iter = 25, 
                                 scoring = 'rmse', 
                                 error_score = 0, verbose = 3, n_jobs = -1)

        elif self.problem_type == 'classification':
            model = XGBClassifier()
            model = RandomizedSearchCV(model_xbg,
                                 param_distributions = param_dist, 
                                 n_iter = 25, 
                                 scoring = 'f1', 
                                 error_score = 0, verbose = 3, n_jobs = -1)

        else:
            raise Exception("Problem Type not supported")

        model.fit(self.training_data.drop(['TRAINING'], axis = 1), self.training_data['TARGET'])

        return model.best_params_

    def _lightgbm(self):
        
        if self.problem_type == 'regression':
            estimator = lgb.LGBMRegressor(num_leaves = 30)

        elif self.problem_type == 'classification':
            estimator = lgb.LGBMClassifier(num_leaves = 30)

        param_grid = {
                'learning_rate': [0.01, 0.1, 1],
                'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500],
        }

        lgb = RandomizedSearchCV(estimator, param_grid, cv=3)
        lgb.fit(self.training_data.drop(['TRAINING'], axis = 1), self.training_data['TARGET'])

        return lgb.best_params_

    def _catboost(self):
        
        if self.problem_type == 'regression':
            model = CatBoostRegressor()
        elif self.problem_type == 'classification':
            model = CatBoostRegressor()
        else:
            raise Exception('Problem Type Not supported!')

        model.fit(self.training_data.drop(['TRAINING'], axis = 1), self.training_data['TARGET'])

        grid = {'learning_rate': [0.03, 0.1],
                'depth': [4, 6, 10],
                'l2_leaf_reg': [1, 3, 5, 7, 9]}

        randomized_search_result = model.randomized_search(grid,
                                                   X=self.training_data.drop(['TARGET'], axis = 1),
                                                   y=self.training_data['TARGET'])
        
        return randomized_search_result['params']
        

    def get_params(self):

        if self.model = "xgboost":
            return self._xgboost()

        elif self.model = "lightgbm":
            return self._lightgbm()

        elif self.model = "catboost":
            return self._catboost()

        else:
            raise Exception("Model not supported")

if __name__ == "__main__":

    Tuner = TuneParams()

    params = Tuner.get_params()

