#improved version of Python_Trading_v5
import datetime
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader._utils
from pandas_datareader import data as wb
import asyncio

class Stock:
    def __init__(self,name):
        self.name = name
        self.analysis_period = 100
        self.data = pd.DataFrame

    async def get_data(self, date = datetime.date.today()):
        start_date = date - datetime.timedelta(days=self.analysis_period)
        self.data = wb.DataReader(f"{self.name}", "yahoo", start_date, date)

    def set_data(self, new_dataframe):
        self.data = new_dataframe


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






async def loop():
    stocks = create_stocks_dict()
    for ticker in stocks.keys():
        try:
            asyncio.create_task(stocks[ticker].get_data())
        except pandas_datareader._utils.RemoteDataError:
            print('Error here')
            stocks[ticker].set_data(new_dataframe=None)

async def pull_data():



    print(stocks[list(stocks.keys())[0]].data)
    return stocks


async def main():
    stocks = await pull_data()
    stocks_keylist = list(stocks.keys())
    print(stocks[stocks_keylist[0]].data)
    print(stocks)









if __name__ == '__main__':
    start = time.time()
    asyncio.run(main())
    print('RUN 20 tickers')
    print(time.time()-start)
