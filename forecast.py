import portfolio_class
import stock_class


from sklearn.ensemble import RandomForestRegressor
import tensorflow_decision_forests as tfdf


def random_forest_sklearn(X,y,n_estimators=100,min_samples_leaf=20, n_jobs=-2, **kwargs):
    """See kwargs at https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html"""
    model = RandomForestRegressor(n_estimators, min_samples_leaf, n_jobs, kwargs)
    model.fit(X,y)
    return model

def regression_tree_sklearn():
    pass

def random_forest_keras(training_data, label_str, num_trees=100, num_threads = 7, min_examples=20,**kwargs ):
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

def xgboost():
    """xgboost.train supports custom objective functions, xgboost.cv supports cross validation
    https://xgboost.readthedocs.io/en/stable/python/python_api.html"""
    pass

def neural_net():
    pass