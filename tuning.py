"""Calculate and backtest a strategy quickly. Good for tuning a specific strategy such as technical_levels."""

import datetime
import random
import matplotlib.pyplot as plt
import stock_class
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import math
import cProfile
import pstats


def tune_tech_levels(from_date, to_date, stocks, param_space, search="grid"):
    """arguments:
    stocks: a list of Stock instances selected by hand or using Stock.random_sample_stocks() function
    param_space: a dictionary of parameters to tune, key: parameter name, value: list of values to consider
    search: search method, one of ('grid', 'random'), if random then the 3rd of total combinations is searched randomly"""
    if not isinstance(param_space, dict):
        raise TypeError('param_space should be a dictionary')
    if not isinstance(stocks, list):
        raise TypeError('stocks should be a list')
    if not isinstance(stocks[0], stock_class.Stock):
        raise TypeError('"stocks" should contain Stock instances')
    if not all([isinstance(dict_val, list) for dict_val in param_space.values()]):
        raise ValueError('all param_space values should be lists')
    if search not in ['grid', 'random']:
        raise ValueError('search should be one of "grid", "random"')

    num_keys = int(len(param_space))
    counter = 1
    combo = list(param_space.values())[0]
    while counter < num_keys:
        combo = combine_lists(combo, list(param_space.values())[counter])
        counter += 1
    if len(param_space) > 2:
        for idx in range(len(combo)):
            combo[idx] = flatten(combo[idx], int(len(param_space) - 2))

    if search == 'random':
        random.shuffle(combo)
        combo = combo[::3]

    result = []
    for params in combo:
        user_input = []
        for stock in stocks:
            if not isinstance(params, (list, tuple)):
                params = [params]
            user_input.append(params)
            param_dict = dict(zip(param_space.keys(), params))
            plt.ion()
            stock.show(from_date, to_date, show_tech_levels=True, **param_dict)
            plt.pause(0.001)
            user_input.append(str(input(f'Press "g" for good and "b" for bad')))
            print(f'{param_dict} -- {user_input}')
            plt.close()
        result.append(user_input)
    return result


def count_good(user_input, keys):
    """counts the good values returned by tune_tech_levels"""
    user_good = [elem for elem in user_input if elem[1] == 'g']
    stripped_g = [lst[0] for lst in user_good]
    df = pd.DataFrame(stripped_g, columns=keys)
    result = []
    for col in df.columns:
        result.append(df.groupby(col)[col].count())
    return result


def combine_lists(lst1, lst2):
    if len(lst1) >= len(lst2):
        lst1 = lst1 * len(lst2)
        lst1 = sorted(lst1)
        lst2 = lst2 * int(len(lst1) / len(lst2))
        return list(zip(lst1, lst2))
    elif len(lst1) < len(lst2):
        lst2 = lst2 * len(lst1)
        lst2 = sorted(lst2)
        lst1 = lst1 * int(len(lst2) / len(lst1))
        return list(zip(lst1, lst2))


def flatten(list_of_lists, num_iter):
    if not isinstance(num_iter, int):
        raise ValueError('num_iter should be int')
    if num_iter == 0:
        return list_of_lists
    else:
        flat_list = []
        i = 0
        while i < num_iter:
            if flat_list:
                list_of_lists = flat_list.copy()
                flat_list = []

            for sublist in list_of_lists:
                # sublist is indeed a list: [(1,2),(3,4)]
                if isinstance(sublist, (list, tuple)):
                    for item in sublist:
                        flat_list.append(item)
                # sublist is actually just a number: [(1,2),3]
                elif isinstance(sublist, (int, float)):
                    flat_list.append(sublist)
            i += 1
        return flat_list


def stationary_maker(input_series, p_val=0.05, maxlag=5, verbose=False):
    """makes an input_series stationary by differencing no more than 2 times and retruns the stationary series

    stationarity is checked with adfuller test using p_val as threshold
    H0: potetntial unit root / non-stationary -- test statistic >= p_val
    H1: no unit root / is stationary -- test statistic < p_val

    args:
    input_series: a pd.Series or a list object
    p_val: p-value below which reject H0
    """
    if isinstance(input_series, (pd.DataFrame, pd.Series)):
        # input series is almost or totally constant over the period
        if len(set(input_series)) < 5:
            if verbose:
                print('constant series, no differencing')
            return input_series

        if adfuller(input_series, maxlag=maxlag)[1] < p_val:
            if verbose:
                print('no differencing')
            return input_series
        elif adfuller(input_series, maxlag=maxlag)[1] >= p_val:
            first_diff = input_series.diff().dropna()
            if adfuller(first_diff, maxlag=maxlag)[1] < p_val:
                if verbose:
                    print('first differencing')
                return first_diff
            elif adfuller(first_diff, maxlag=maxlag)[1] >= p_val:
                second_diff = first_diff.diff().dropna()
                if verbose:
                    print('second differencing')
                return second_diff

    elif isinstance(input_series, list):
        input_series = pd.Series(input_series)
        stationary_maker(input_series, p_val)
    else:
        raise TypeError(f"invalid type {type(input_series)}")


def is_stationary(input_series, p_val=0.05, maxlag=5):
    if adfuller(input_series, maxlag=maxlag)[1] < p_val:
        return True
    else:
        return False

def plot_acf(input_series, alpha=0.05, auto_ylim=True):
    if not isinstance(input_series, pd.Series):
        input_series = pd.Series(input_series)
    plot_acf(input_series.dropna(), lags=[1,2,3,4,5,6,7,8,9,10], alpha=alpha, auto_ylims=auto_ylim)

def get_outliers(input_series, multiplier=3.69, verbose=True):
    """identifies outliers in a series
       steps:
       1)makes the series stationary if not already
       2)outlier is defined as outside the [mean - (std * multiplier) , mean + (std * multiplier)]

       args:
       input_series: list/pd.Series/np.array
       multiplier: float, default=3.69, z-value for 0.001 (0.1%)
       """
    if isinstance(input_series, pd.Series):
        plot_title = input_series.name
    else:
        plot_title = None
        input_series = pd.Series(input_series)

    if is_stationary(input_series):
        pass
    else:
        input_series = stationary_maker(input_series)

    stdev = math.sqrt(np.var(input_series, ddof=1))
    result = pd.DataFrame({'series': input_series, 'outliers': np.zeros((len(input_series)))})
    upper_bound = input_series.mean()+(multiplier*stdev)
    lower_bound = input_series.mean()-(multiplier*stdev)
    result['outliers'].where(result['series'].between(lower_bound, upper_bound, inclusive='neither'), 1, inplace=True)
    if verbose:
        plt.figure(figsize=(16, 6))
        if plot_title:
            plt.title(plot_title)
        plt.plot(result['series'], label='original')
        plt.plot(result.loc[result['outliers'] == 0, 'series'], label='outliers removed')
        plt.legend()
    return result



def closest_number(num, lst):
    """returns the element from lst closest (in absolute value) to num"""
    try:
        if isinstance(lst[0], (tuple, list)):
            lst = flatten(lst, 1)
    except IndexError:
        return None
    curr = lst[0]
    for elem in lst:
        if abs(num-elem) < abs(num-curr):
            curr = elem
    return curr


def profiler(func, **kwargs):
    with cProfile.Profile() as pr:
        result = func(**kwargs)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    return result

