import concurrent.futures
import datetime
from pandas_datareader import data as wb
import pandas as pd
import numpy as np
import database
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Constants
MAX_WORKERS = 20
LOOK_AHEAD_RANGE = 31
PLACEHOLDER = 10


class Stock:
    # stock_list contains Stock instances not just names
    stock_list = []
    yahoo_pull_start_date = datetime.date(2007, 1, 9)
    yahoo_pull_end_date = datetime.date(2007, 1, 19)

    def __init__(self, name):
        # stock related attributes
        self.name = name
        self.data = pd.DataFrame
        self.sector = str
        # portfolio related attributes
        self.log = pd.DataFrame({'date':[np.nan]*PLACEHOLDER,
                                  'amount': [np.nan]*PLACEHOLDER,
                                  'price': [np.nan]*PLACEHOLDER,
                                  'value': [np.nan]*PLACEHOLDER})
        self.buy_log = pd.DataFrame({'date':[np.nan]*PLACEHOLDER,
                                  'amount': [np.nan]*PLACEHOLDER,
                                  'price': [np.nan]*PLACEHOLDER,
                                  'value': [np.nan]*PLACEHOLDER})
        self.sell_log = pd.DataFrame({'date':[np.nan]*PLACEHOLDER,
                                  'amount': [np.nan]*PLACEHOLDER,
                                  'price': [np.nan]*PLACEHOLDER,
                                  'value': [np.nan]*PLACEHOLDER})
        self.current_amount = float
        self.current_value = float

        # prevent multiple initialization
        names = Stock.get_stock_names()
        if self.name not in names:
            self.__class__.stock_list.append(self)


    def __str__(self):
        print("Stock:")
        print(f"Name: {self.name}")

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
    def clear_stock_list(cls):
        cls.stock_list = []

    @classmethod
    def yahoo_pull_data_for_stock_list(cls):
        if len(cls.stock_list) > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(ticker.yahoo_pull_data) for ticker in cls.stock_list]
                concurrent.futures.wait(futures)

        return futures

    @classmethod
    def pop_no_data_tickers(cls):
        drop = [name for name in Stock.get_stock_names() if Stock.fetch_stock(name).data.empty]
        cls.stock_list = [item for item in cls.stock_list if not item.data.empty]
        print(f"No data available: {drop}")

    @classmethod
    def fetch_stock(cls, name):
        for stock in cls.stock_list:
            if stock.name == name:
                return stock
        raise ValueError(f'stock_list does not contain {name}')


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
            raise TypeError('new_dataframe: not pandas.dataframe')
        new_dataframe.rename(columns={'date_':'Date', 'close':'Close', 'ticker':'Ticker'}, inplace=True)
        new_dataframe.set_index(new_dataframe['Date'], inplace=True)
        self.data = new_dataframe

    def set_sector(self):
        #get sector for one particular stock from sql database
        pass

    def set_yahoo_pull_start_date(self, new_date):
        self.yahoo_pull_start_date = new_date

    def set_yahoo_pull_end_date(self, new_date):
        self.yahoo_pull_end_date = new_date

    def labeling_function(self, method = 'minmax'):
        look_ahead_range = LOOK_AHEAD_RANGE
        label = []

        if method not in ['minmax', 'max', 'avg']:
            raise ValueError("Choose a correct method: 'minmax', 'max', 'avg'")

        for close_idx in range(len(self.data['Close'])):
            relative_profits = \
                self.data['Close'].iloc[(close_idx+1) : close_idx + look_ahead_range]/self.data['Close'].iloc[close_idx]

            # minus 1: so that a 4% decrease is not represented as 96% but as -4%
            # this leads to a plus in the minmax method
            relative_profits = relative_profits - 1

            if method == 'minmax':
                if not len(relative_profits) == 0:
                    label.append(max(relative_profits) + (2 * min(relative_profits)))
                else:
                    label.append(np.nan)
            elif method == 'max':
                if not len(relative_profits) == 0:
                    label.append(max(relative_profits))
                else:
                    label.append(np.nan)
            elif method == 'avg':
                label.append(relative_profits.mean())

        if len(label) < len(self.data['Close']):
            diff_length = len(self.data['Close'])-len(label)
            tmp_array = np.empty(diff_length)
            tmp_array[:] = np.nan
            label.append(list(tmp_array))

        self.data[f'label_{method}'] = label

        # Other approach:
        # maxima = self.data['Close'][::-1].rolling(window=look_ahead_range).max()[::-1]
        # minima = self.data['Close'][::-1].rolling(window=look_ahead_range).min()[::-1]
        # self.data['label'] = maxima - (2 * minima)

    def sma_calc(self, period=20):
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

    def rsi_calc(self, lookback=14):
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

    def show(self, from_date = datetime.date(2000,1,1), to_date = datetime.date.today()):
        if not isinstance(from_date, datetime.date):
            from_date = datetime.date(from_date)
        if not isinstance(to_date, datetime.date):
            to_date = datetime.date(to_date)
        chunk = self.data.loc[from_date : to_date]
        plt.plot(chunk['Close'])
        plt.show()

    def get_price(self, date):
        try:
            return self.data.loc[date, 'Close']
        except TypeError:
            print(f'Stock.data is not set for {self.name}. First fill it from yahoo or sql.')


    def get_price_range(self, from_date = datetime.date(2000,1,1), to_date = datetime.date.today()):
        try:
            return self.data.loc[from_date : to_date]
        except TypeError:
            print(f'Stock.data is not set for {self.name}. First fill it from yahoo or sql.')

    def get_technical_levels(self,
                             from_date = datetime.date(2000,1,1),
                             to_date = datetime.date.today(),
                             distance = 1,
                             threshold = 0,
                             width = 0.005):
        data_chunk = self.get_price_range(from_date, to_date)
        print('data_chunk',data_chunk)
        peaks, _ = find_peaks(data_chunk['Close'], distance=distance, threshold=threshold)
        print('peaks', peaks)
        troughs, _ = find_peaks(data_chunk['Close'] * (-1), distance=distance, threshold=threshold)
        peaks_troughs_df = data_chunk.loc[self.data['Date'][np.concatenate((peaks, troughs))], ('Date', 'Close')]
        peaks_troughs_df = peaks_troughs_df.sort_values(by='Close')
        print(peaks_troughs_df)

        # if the next price level is within 'width' add it to the ith technical level
        # don't care if added multiple times -- convert tech_levels lower level lists to sets (or only min-max values)
        tech_levels = []
        for i in range(len(peaks_troughs_df['Close'])):
            tech_levels[i] = []
            for k in range(i+1, len(peaks_troughs_df['Close'])):
                if abs(peaks_troughs_df.loc[i, 'Close'] -
                       peaks_troughs_df.loc[k, 'Close']) < width * peaks_troughs_df.loc[i, 'Close']:
                    tech_levels[i].append(peaks_troughs_df.loc[(i,k), 'Close'])

        # dropping empty sublists of tech_levels list-of-lists
        empty = []
        for j in range(len(tech_levels)):
            if not sublist[j]:
                empty.append(j)
        for l in empty:
            tech_levels.pop(l)

        return tech_levels

    def plot_technical_levels(self):
        pass






