#improved version of Python_Trading_v5
import datetime
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader._utils
from pandas_datareader import data as wb
import concurrent.futures


class Stock:
    def __init__(self,name):
        self.name = name
        self.analysis_period = 100
        self.data = pd.DataFrame

    def set_data(self, new_dataframe):
        self.data = new_dataframe


def get_data(dict,key, date = datetime.date.today()):
    start_date = date - datetime.timedelta(days=dict[key].analysis_period)
    dict[key].set_data(wb.DataReader(f"{dict[key].name}", "yahoo", start_date, date))

def create_stocks_dict():
    stocks = {}
    excel_data = get_tickers()
    tickers = excel_data.Symbol.copy()

    for ticker in tickers[:20]:
        stocks[f"{ticker}"] = Stock(name=ticker)

    return stocks



def get_tickers():
    tickers = pd.read_csv(open("nasdaq_tickers.csv",encoding="latin-1"))
    return tickers



def pull_data():
    stocks = create_stocks_dict()

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(get_data, dict=stocks, key=ticker) for ticker in stocks.keys()]
        for item in concurrent.futures.wait(futures):
            print('ended')
    #print(stocks[list(stocks.keys())[0]].data)
    return stocks



def main():
    stocks = pull_data()
    stocks_keylist = list(stocks.keys())
    for i in range(len(stocks.keys())):
        print(stocks[stocks_keylist[i]].data)


if __name__ == '__main__':
    start = time.time()
    main()
    print('RUN 20 tickers')
    print(time.time()-start)
