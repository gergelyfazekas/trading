"""Calculate and backtest a strategy quickly. Good for tuning a specific strategy such as technical_levels."""

import datetime
import random
import matplotlib.pyplot as plt
import stock_class
import numpy as np


def tune_tech_levels(from_date, to_date, stocks, param_space, search = "grid"):
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
            combo[idx] = flatten(combo[idx], int(len(param_space)-2))

    if search == 'random':
        random.shuffle(combo)
        combo = combo[::3]

    result = []
    for params in combo:
        user_input = []
        for stock in stocks:
            if not isinstance(params, list):
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
        i=0
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
            i+=1
        return flat_list



