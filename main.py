#improved version of Python_Trading_v5
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader._utils
from pandas_datareader import data as wb
import concurrent.futures


ANALYSIS_PERIOD = 20
NUMBER_OF_TICKERS = 20

MAX_WORKERS = min(20,NUMBER_OF_TICKERS)


class Stock:
    def __init__(self,name):
        self.name = name
        self.analysis_period = ANALYSIS_PERIOD
        self.data = pd.DataFrame

    def set_data(self, new_dataframe):
        self.data = new_dataframe


def get_data(dict,key, date = datetime.date.today()):
    start_date = date - datetime.timedelta(days=dict[key].analysis_period)
    dict[key].set_data(wb.DataReader(f"{dict[key].name}", "yahoo", start_date, date))


def create_stocks_dict():
    stocks = {}
    excel_data = read_tickers_from_csv()
    tickers = excel_data.Symbol.copy()

    for ticker in tickers[:NUMBER_OF_TICKERS]:
        stocks[f"{ticker}"] = Stock(name=ticker)

    return stocks


def read_tickers_from_csv():
    tickers = pd.read_csv(open("nasdaq_tickers.csv",encoding="latin-1"))
    return tickers


def pull_data():
    stocks = create_stocks_dict()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(get_data, dict=stocks, key=ticker) for ticker in stocks.keys()]
        concurrent.futures.wait(futures)

    stocks_keylist = list(stocks.keys())
    no_data_tickers = [stocks_keylist[x] for x in range(len(stocks_keylist)) if stocks[stocks_keylist[x]].data.empty]
    pop_keys(dict=stocks, keys_to_pop=no_data_tickers)
    print("No data found (and removed from dict): ", no_data_tickers)

    return stocks


def pop_keys(dict, keys_to_pop):
    for key in keys_to_pop:
        dict.pop(key)
    return dict


def main():
    stocks = pull_data()
    stocks_keylist = list(stocks.keys())

    # for i in range(len(stocks.keys())):
    #     print(stocks[stocks_keylist[i]].data)


if __name__ == '__main__':
    print("started")
    start = time.time()

    main()

    print("tottime: ",time.time()-start)
