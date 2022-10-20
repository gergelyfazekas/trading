import stock_class


from sklearn.ensemble import RandomForestRegressor
# import tensorflow_decision_forests as tfdf
import xgboost as xgb


def random_forest_sklearn(training_data, label_str, n_estimators=20,min_samples_leaf=20, n_jobs=-2, **kwargs):
    """See kwargs at https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html"""
    model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs=n_jobs, **kwargs)
    model.fit(training_data.loc[:, training_data.columns != label_str], training_data.loc[:, label_str])
    return model

def regression_tree_sklearn():
    pass

def random_forest_keras(training_data, label_str, num_trees=20, num_threads = 7, min_examples=20,**kwargs ):
    """See kwargs and hyperparameter tuning at
    https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel#predict"""

    tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(training_data, label=label_str)
    model = tfdf.keras.RandomForestModel(num_trees, num_threads, min_examples)
    model.fit(tf_dataset)
    return model

def regression_tree_keras():
    pass

def boosting_sklearn():
    pass

def boosting_keras():
    pass

def xgbooster(train_txt_path, num_round, param_dict=None):
    """xgboost.train supports custom objective functions, xgboost.cv supports cross validation
    https://xgboost.readthedocs.io/en/stable/python/python_api.html,
    sample code: https://xgboost.readthedocs.io/en/stable/get_started.html"""
    dtrain = xgb.DMatrix(train_txt_path)
    if not param_dict:
        param_dict = {'max_depth':2, 'eta':1, 'objective':'reg:squarederror'}
    model = xgb.train(param_dict, dtrain, num_round)
    return model

def neural_net():
    pass