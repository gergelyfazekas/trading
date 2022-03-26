import concurrent.futures
import datetime
from pandas_datareader import data as wb
import pandas as pd
import numpy as np
import database
import sys

# Constants
MAX_WORKERS = 20
CALCULATION_PERIOD = 20


class Stock:
    # stock_list contains Stock instances not just names
    stock_list = []
    yahoo_pull_start_date = datetime.date(2007,1,9)
    yahoo_pull_end_date = datetime.date(2007,1,10)

    @staticmethod
    def create_top_tickers_csv(filename="nasdaq_tickers.csv", number=30, sort_by="Market Cap", asc=False):
        tickers = pd.read_csv(open(filename, encoding="latin-1"))
        tickers = tickers.sort_values(by=sort_by, ascending=asc)
        tickers_by_ipo = tickers.loc[(pd.notnull(tickers['IPO Year'])) & (tickers['IPO Year'] <= 2006)]
        top_tickers = tickers_by_ipo.iloc[0:number, :]
        top_tickers.to_csv(f"tickers_{number}.csv")
        print(f"tickers_{number}.csv was created")

    @classmethod
    def create_stock_list_from_csv(cls, filename="tickers_30.csv"):
        excel_data = pd.read_csv(open(filename, encoding="latin-1"))
        tickers = excel_data['Symbol'].copy()
        tickers = tickers.values.tolist()
        for ticker in tickers:
            Stock(name=ticker)

    @classmethod
    def print_stock_names(cls):
        stock_names_lst = []
        for stock in cls.stock_list:
            stock_names_lst.append(stock.name)
        print(stock_names_lst)

    @classmethod
    def get_stock_names(cls):
        ticker_lst = []
        for stock in cls.stock_list:
            ticker_lst.append(stock.name)
        return ticker_lst

    @classmethod
    def yahoo_pull_data_for_stock_list(cls):
        if len(cls.stock_list) > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(ticker.yahoo_pull_data) for ticker in cls.stock_list]
                concurrent.futures.wait(futures)

        return futures

            # print(type(futures[1].result()))
            # no_data_tickers = [item.name for item in cls.stock_list if
            #                    item.data.empty]

            # cls.pop_no_data_tickers()
            # print("No data found (and removed from dict): ", no_data_tickers)

    @classmethod
    def pop_no_data_tickers(cls):
        cls.stock_list = [item for item in cls.stock_list if not item.data.empty]

    def __init__(self, name):
        self.name = name
        self.__class__.stock_list.append(self)
        self.calculation_period = CALCULATION_PERIOD
        self.data = pd.DataFrame

    def yahoo_pull_data(self):
        yahoo_data = wb.DataReader(self.name, "yahoo", self.yahoo_pull_start_date, self.yahoo_pull_end_date)
        if not isinstance(yahoo_data, pd.DataFrame):
            raise TypeError('yahoo_data: not pandas.dataframe')
        yahoo_data["Date"] = yahoo_data.index.strftime('%Y-%m-%d %X')
        yahoo_data["Ticker"] = self.name
        return yahoo_data

    def set_data(self, new_dataframe):
        if not isinstance(new_dataframe, pd.DataFrame):
            print(new_dataframe)
            raise TypeError('new_dataframe: not pandas.datafram e')
        self.data = new_dataframe

    def set_yahoo_pull_start_date(self, new_date):
        self.yahoo_pull_start_date = new_date

    def set_yahoo_pull_end_date(self, new_date):
        self.yahoo_pull_end_date = new_date

    def sma_calc(self, period=CALCULATION_PERIOD):
        self.data[f'sma_{period}'] = self.data['Close'].rolling(window=period).mean()

    def sma_cross(self, sma_short_days, sma_long_days):
        if not f'sma_{sma_long_days}' in self.data.columns:
            self.sma_calc(period=sma_long_days)
        if not f'sma_{sma_short_days}' in self.data.columns:
            self.sma_calc(period=sma_short_days)

        below_or_above = np.where(self.data[f'sma_{sma_short_days}'] < self.data[f'sma_{sma_long_days}'], 0,
                                  np.where(self.data[f'sma_{sma_short_days}'] > self.data[f'sma_{sma_long_days}'], 1,
                                           np.nan))

        crosses = [0]
        for i in range(len(below_or_above) - 1):
            crosses.append(below_or_above[i + 1] - below_or_above[i])

        self.data[f'sma_cross_{sma_short_days}_{sma_long_days}'] = crosses

    def rsi(self, lookback):
        values = self.data['Close'].iloc[-lookback:]
        diffs = [values[x+1]-values[x] for x in range(len(values)-1)]
        ups = diffs[diffs > 0].mean()
        downs = -1 * diffs[diffs < 0].mean()
        # print('rsi', self.name, 100 * up / (up + down))
        self.data[f'rsi_{lookback}'] = 100 * ups / (ups + downs)