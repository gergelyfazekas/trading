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
LOOK_AHEAD_RANGE = 31


class Stock:
    # stock_list contains Stock instances not just names
    stock_list = []
    yahoo_pull_start_date = datetime.date(2007, 1, 9)
    yahoo_pull_end_date = datetime.date(2007, 1, 10)

    def __init__(self, name):
        self.name = name
        self.__class__.stock_list.append(self)
        self.calculation_period = CALCULATION_PERIOD
        self.data = pd.DataFrame

    def __str__(self):
        return f"Stock:" \
               f"Name: {self.name}"

    def __repr__(self):
        return f"Stock(name={self.name})"

    def __eq__(self, other):
        if self.name == other.name and len(self.data.index) == len(other.data.index) and np.all(
                self.data.index == other.data.index):
            return True
        else:
            return False

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

    @classmethod
    def pop_no_data_tickers(cls):
        cls.stock_list = [item for item in cls.stock_list if not item.data.empty]

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

    def labeling_function(self):
        look_ahead_range = LOOK_AHEAD_RANGE
        label = []
        for close_idx in range(len(self.data['Close'])):
            relative_profit = []
            for look in range(len(look_ahead_range)):
                # This will raise IndexError when overloading - Need to fix this later
                relative_profit.append(self.data['Close'][close_idx + look] / self.data['Close'][close_idx])

            label.append(max(relative_profit) - (2 * min(relative_profit)))

        if len(label) < len(self.data['Close']):
            diff_length = len(self.data['Close'])-len(label)
            tmp_array = np.empty(diff_length)
            tmp_array[:] = np.nan
            label.append(list(tmp_array))
            
        self.data['label'] = label

        # Other approach:
        # maxima = self.data['Close'][::-1].rolling(window=look_ahead_range).max()[::-1]
        # minima = self.data['Close'][::-1].rolling(window=look_ahead_range).min()[::-1]
        # self.data['label'] = maxima - (2 * minima)

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

    def rsi_calc(self, lookback):
        # values = self.data['Close'].iloc[-lookback:]
        # diffs = [values[x+1]-values[x] for x in range(len(values)-1)]
        # diffs = pd.Series(diffs)

        diffs = self.data['Close'].diff()
        ups = diffs.where(diffs > 0, 0)
        downs = -1 * diffs.where(diffs < 0, 0)
        up_sma = ups.rolling(window=lookback).mean()
        down_sma = downs.rolling(window=lookback).mean()
        rs_factor = up_sma / down_sma
        self.data[f'rsi_{lookback}'] = 100 - (100 / (1 + rs_factor))
