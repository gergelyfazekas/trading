import concurrent.futures
import datetime
import matplotlib.pyplot
from pandas_datareader import data as wb
import pandas as pd
import numpy as np
import database
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Constants
START_DATE = datetime.date(2000, 1, 1)
END_DATE = datetime.date(2000, 1, 10)
MAX_WORKERS = 20
LOOK_AHEAD_RANGE = 31
PLACEHOLDER = 1000


class Stock:
    # stock_list contains Stock instances not just names
    stock_list = []

    def __init__(self, name):
        # stock related attributes
        self.name = name
        self.data = pd.DataFrame
        self.sector = str
        self.yahoo_pull_start_date = START_DATE
        self.yahoo_pull_end_date = END_DATE
        # portfolio related attributes
        self.log = pd.DataFrame({'date_': [np.nan] * PLACEHOLDER,
                                 'amount': [np.nan] * PLACEHOLDER,
                                 'price': [np.nan] * PLACEHOLDER,
                                 'value': [np.nan] * PLACEHOLDER})
        self.buy_log = pd.DataFrame({'date_': [np.nan] * PLACEHOLDER,
                                     'amount': [np.nan] * PLACEHOLDER,
                                     'price': [np.nan] * PLACEHOLDER,
                                     'value': [np.nan] * PLACEHOLDER})
        self.sell_log = pd.DataFrame({'date_': [np.nan] * PLACEHOLDER,
                                      'amount': [np.nan] * PLACEHOLDER,
                                      'price': [np.nan] * PLACEHOLDER,
                                      'value': [np.nan] * PLACEHOLDER})
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
    def create_stock_list_sql(cls, ticker_list):
        for ticker in ticker_list:
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
        drop = [name for name in cls.get_stock_names() if cls.fetch_stock(name).data.empty]
        cls.stock_list = [item for item in cls.stock_list if not item.data.empty]
        print(f"No data available: {drop}")

    @classmethod
    def fetch_stock(cls, name):
        for stock in cls.stock_list:
            if stock.name == name:
                return stock
            else:
                raise KeyError(f'stock_list does not contain {name}')

    @classmethod
    def random_sample_stocks(cls, num_draws):
        if len(cls.stock_list) == 1:
            return stock_list
        if len(cls.stock_list) < num_draws:
            raise ValueError("num_draws larger than stock_list")
        idx_lst = np.random.randint(0, len(cls.stock_list) - 1, num_draws)
        drawn_stocks = []
        for idx in idx_lst:
            drawn_stocks.append(cls.stock_list[idx])
        return drawn_stocks

    @classmethod
    def strategy_looper_single_param(cls, strategy_dict):
        """calculates the strategies for every stock in stock_list and pushes it back to sql (NOT IMPLEMENTED YET)
        every strategy should be single parameter so that:
            if [1,5,10] is given then the strategy is called three times with 1, then 5, then 10 as parameter
            the strategies are implemented so that a new column is created at each call (see sma_calc)

        arguments:
        strategy_dict: key is the name of the strategy, value is a list of parameters to calculate
        push_back: NOT IMPLEMENTED YET if True push back the extended dataframe to sql
        """
        for stock in cls.stock_list:
            for strategy, params in strategy_dict.items():
                for param in params:
                    try:
                        strategy(stock, param)
                    except ValueError as err:
                        print(err)
                        continue

    @classmethod
    def aggregate_data(cls):
        col_names = cls.stock_list[0].data.columns
        total_df = pd.DataFrame(columns=col_names)
        for stock in cls.stock_list:
            total_df = pd.concat([total_df, stock.data])
        return total_df

    def yahoo_pull_data(self):
        yahoo_data = wb.DataReader(self.name, "yahoo", self.yahoo_pull_start_date, self.yahoo_pull_end_date)
        if not isinstance(yahoo_data, pd.DataFrame):
            raise TypeError('yahoo_data: not pandas.dataframe')
        yahoo_data['ticker'] = self.name
        self.set_data(yahoo_data)
        return self.data

    def lowercase(self):
        uppercase_names = self.data.columns
        lowercase_names = [old_name.lower() for old_name in self.data.columns]
        self.data.rename(columns=dict(zip(uppercase_names, lowercase_names)), inplace=True)

    def set_index(self):
        if self.data.index.name != 'date_':
            if self.data.index.name != 'Date':
                if not 'date_' in self.data.columns:
                    try:
                        self.data['date_'] = self.data['Date']
                    except KeyError:
                        raise ('No "date_" or "Date" column')
                else:
                    self.data.set_index('date_', inplace=True)
            else:
                self.data.index.name = 'date_'

        self.data.index = pd.to_datetime(self.data.index).date
        self.data['date_'] = self.data.index.copy()
        self.data.rename(columns={'adj close': 'adj_close'}, inplace=True)
        self.data.rename(columns={'Adj Close': 'adj_close'}, inplace=True)

    def set_data(self, new_dataframe):
        if not isinstance(new_dataframe, pd.DataFrame):
            print(new_dataframe)
            raise TypeError('new_dataframe: not pandas.dataframe')
        self.data = new_dataframe

    def set_sector(self):
        pass

    def set_yahoo_pull_start_date(self, new_date):
        self.yahoo_pull_start_date = new_date

    def set_yahoo_pull_end_date(self, new_date):
        self.yahoo_pull_end_date = new_date

    def labeling_function(self, method='minmax'):
        look_ahead_range = LOOK_AHEAD_RANGE
        label = []

        if method not in ['minmax', 'max', 'avg']:
            raise ValueError("Choose a correct method: 'minmax', 'max', 'avg'")

        for close_idx in range(len(self.data['close'])):
            relative_profits = \
                self.data['close'].iloc[(close_idx + 1): close_idx + look_ahead_range] / self.data['close'].iloc[
                    close_idx]

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
        # padding 
        if len(label) < len(self.data['close']):
            diff_length = len(self.data['close']) - len(label)
            padding = np.array([np.nan] * diff_length)
            label.append(list(padding))

        self.data[f'label_{method}'] = label

    def sma_calc(self, period=20):
        """calculates simple moving average with window length=period
        creates a new column for the sma_{period} in self.data
        if data is too short for given period a ValueError is raised
        """
        if len(self.data.index) <= period:
            raise ValueError(f'len(self.data.index)={len(self.data.index)} for {self.name} is shorter than period={period}')
        if f'sma_{period}' not in self.data.columns:
            self.data[f'sma_{period}'] = self.data['close'].rolling(window=period).mean()
        else:
            print(f'sma_{period} already exists')

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
        diffs = self.data['close'].diff()
        ups = diffs.where(diffs > 0, 0)
        downs = -1 * diffs.where(diffs < 0, 0)
        up_sma = ups.rolling(window=lookback).mean()
        down_sma = downs.rolling(window=lookback).mean()
        rs_factor = up_sma / down_sma
        self.data[f'rsi_{lookback}'] = 100 - (100 / (1 + rs_factor))

    def show(self, from_date=datetime.date(2000, 1, 1), to_date=datetime.date.today(), show_tech_levels=False,
             **kwargs):
        """kwargs:
        1)tech_width -- own argument determining the width of a tech_level
        2)other kwargs passed to scipy.signal.find_peaks"""
        if not isinstance(from_date, datetime.date):
            try:
                from_date = datetime.date(from_date)
            except TypeError:
                print(f'type of from_date: {type(from_date)}')
        if not isinstance(to_date, datetime.date):
            try:
                to_date = datetime.date(to_date)
            except TypeError:
                print(f'type of to_date: {type(from_date)}')
        chunk = self.data.loc[from_date: to_date]
        first_date = chunk.first_valid_index()
        last_date = chunk.last_valid_index()
        plt.plot(chunk['close'])

        if show_tech_levels:
            tech_levels = self.get_technical_levels(from_date=first_date, to_date=last_date, **kwargs)
            y_coords = [sum(sublist) / len(sublist) for sublist in tech_levels]
            plt.hlines(y_coords, xmin=first_date, xmax=last_date)
        plt.show()

    def get_price(self, as_of):
        try:
            return self.data.loc[as_of, 'close']
        except TypeError:
            print(f'Stock.data is not set for {self.name}. First fill it from yahoo or sql.')

    def get_price_range(self, from_date=datetime.date(2000, 1, 1), to_date=datetime.date.today()):
        try:
            return self.data.loc[from_date: to_date]
        except TypeError:
            print(f'Stock.data is not set for {self.name}. First fill it from yahoo or sql.')

    def get_high_volumes(self, from_date=datetime.date(2000, 1, 1), to_date=datetime.date.today(),
                         auto=True,
                         number=None,
                         **kwargs):
        """arguments:
        auto: if True the top 10 volumes plus the find_peaks(volumes, **kwargs) are returned
        number: if not None then the top number of volumes are returned from the set found by auto
        kwargs: passed to scipy.signal.find_peaks()"""

        data_chunk = self.get_price_range(from_date, to_date)
        auto_size = min(10, len(data_chunk['volume']))
        max_dates_idx = pd.Series(data_chunk['volume'].sort_values()[-auto_size:].index)
        peak_vol_idx, _ = find_peaks(data_chunk['volume'], **kwargs)
        peak_vol_dates = data_chunk['date_'][peak_vol_idx]
        unique_dates = peak_vol_dates.append(max_dates_idx, ignore_index=True)
        peaks_df = data_chunk.loc[self.data['date_'][unique_dates.drop_duplicates()], ('date_', 'volume')]
        return peaks_df

    def get_technical_levels(self, from_date=datetime.date(2000, 1, 1), to_date=datetime.date.today(), **kwargs):
        """kwargs:
        1)tech_width -- own argument determining the width of a level
        2)consider_volume = True
        2)other kwargs passed to scipy.signal.find_peaks"""
        try:
            tech_width = kwargs['tech_width']
            kwargs.pop('tech_width')
        except KeyError:
            tech_width = 0.005

        try:
            consider_volume = kwargs['consider_volume']
            kwargs.pop('consider_volume')
        except KeyError:
            consider_volume = True

        try:
            kwargs['rel_height']
        except KeyError:
            kwargs['rel_height'] = 0.5

        data_chunk = self.get_price_range(from_date, to_date)
        peaks, _ = find_peaks((data_chunk['close'] / data_chunk['close'][-1]), **kwargs)
        troughs, _ = find_peaks((data_chunk['close'] / data_chunk['close'][-1]) * (-1), **kwargs)

        if consider_volume:
            high_vol_df = self.get_high_volumes(from_date, to_date)

        high_price_df = data_chunk.loc[
            self.data['date_'][np.concatenate((peaks, troughs))],
            ('date_', 'close')
        ]
        if consider_volume:
            unique_dates = high_vol_df['date_'].append(high_price_df['date_'], ignore_index=True)
            tech_df = data_chunk.loc[unique_dates.drop_duplicates(), ('date_', 'close')]
        # if the next price level is within 'tech_width' add it to the ith technical level
        # don't care if added multiple times (later converts tech_levels sublists to min-max values)
        tech_levels = []
        close_column_idx = tech_df.columns.get_loc('close')
        for i in range(len(tech_df['close'])):
            tech_levels.append([])
            for k in range(i + 1, len(tech_df['close'])):
                if abs(tech_df.iloc[i, close_column_idx] -
                       tech_df.iloc[k, close_column_idx]) \
                        < tech_width * tech_df.iloc[i, close_column_idx]:
                    tech_levels[i].append(tech_df.iloc[[i, k], close_column_idx])

        # dropping empty sublists of tech_levels list-of-lists
        tech_levels = np.array(tech_levels)
        tech_levels = tech_levels[np.nonzero(tech_levels)[0]].tolist()
        tech_levels = [[min(sublist[0]), max(sublist[0])] for sublist in tech_levels]

        return tech_levels
