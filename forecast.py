import stock_class
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
# import tensorflow_decision_forests as tfdf
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate


def ols(training_data, label_str):
    pass

def random_forest_sklearn(training_data, label_str, n_estimators=20,min_samples_leaf=20, n_jobs=-2, **kwargs):
    """See kwargs at https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html"""
    model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs=n_jobs, **kwargs)
    model.fit(training_data.loc[:, training_data.columns != label_str], training_data.loc[:, label_str])
    return model

def regression_tree_sklearn():
    pass

def random_forest_keras(training_data, label_str, num_trees=20, num_threads = 7, min_examples=20, **kwargs):
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


def lasso(training_data, label_str, alpha, **kwargs):
    """computes lasso regression using sklearn.Lasso

    kwargs:
    passed to sklearn.Lasso, e.g. tol, warm_start, selection"""
    model = Lasso(alpha=alpha, **kwargs)
    model.fit(training_data.loc[:, training_data.columns != label_str], training_data.loc[:, label_str])
    return model


def hist_gradient_booster(training_data, label_str, **kwargs):
    """gradient boosting using native NaN handling
    categorical features have to be set in pd.DataFrame as categorical: total_df['sector_encoded'].astype("category")
    **kwargs are passed to HistGradBoost:
        except categorical_features which is created inside this function and should not be passed among kwargs!!
    """
    X = training_data.loc[:, training_data.columns != label_str].copy()
    y = pd.Series(training_data.loc[:, label_str].copy())

    category_mask = list(X.dtypes == "category")
    hgb = HistGradientBoostingRegressor(random_state=42, categorical_features=category_mask, **kwargs)
    model = hgb.fit(X, y)
    return model


def gradient_booster(training_data, label_str, **kwargs):
    """gradient boosting

    possible parameters:
    n_estimators: boosting stages to perform
    subsample: fraction of samples to fit each base learner, values smaller than 1 result in stochastic gradient descent
    min_samples_leaf
    """
    X = training_data.loc[:, training_data.columns != label_str].copy()
    y = pd.Series(training_data.loc[:, label_str].copy())

    categorical_var_names = X.columns[X.dtypes == "category"]
    if not categorical_var_names.empty:
        for col_name in categorical_var_names:
            dummy_df = pd.get_dummies(X[col_name], prefix=col_name)
            X.drop(labels=col_name, axis=1, inplace=True)
            X = pd.concat([X, dummy_df], axis=1)
    hgb = GradientBoostingRegressor(random_state=42, **kwargs)
    try:
        model = hgb.fit(X, y)
    except ValueError:
        X.fillna(100000, inplace=True)
        model = hgb.fit(X, y)
    return model
