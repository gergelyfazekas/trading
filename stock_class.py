import concurrent.futures
import datetime
import matplotlib.pyplot
from pandas_datareader import data as wb
import pandas as pd
import numpy as np
import database
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import tuning
import inspect
import math
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Constants
START_DATE = datetime.date(1990, 1, 1)
END_DATE = datetime.date(2020, 1, 1)
MAX_WORKERS = 20
LOOK_AHEAD_RANGE = 31
PLACEHOLDER = 100000


class Stock:
    # stock_list contains Stock instances not just names
    stock_list = []
    stock_dict = {}

    def __init__(self, name):
        # stock related attributes
        self.name = name
        self.sector = str
        self.yahoo_pull_start_date = START_DATE
        self.yahoo_pull_end_date = END_DATE
        self.data = pd.DataFrame()

        # prevent multiple initialization
        current_frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(current_frame, 2)
        caller_name = caller_frame[1][3]
        # creating stocks one-by-one with Stock('AAPL')
        if caller_name != "create_stock_list":
            if self.name not in self.__class__.get_stock_names():
                self.__class__.stock_list.append(self)
                self.__class__.stock_dict[self.name] = self
            else:
                raise ValueError(f'{self.name} already in stock_list')
        # creating stocks with cls.create_stock_list
        else:
            self.__class__.stock_list.append(self)
            self.__class__.stock_dict[self.name] = self

    @classmethod
    def create_stock_list(cls, ticker_list):
        cls.clear_stock_list()
        ticker_set = set(ticker_list)
        for ticker in ticker_set:
            Stock(name=ticker)

    def __str__(self):
        print("Stock:")
        print(f"Name: {self.name}")

    def __repr__(self):
        return f"Stock(name='{self.name}')"

    def __eq__(self, other):
        return self.name == other.name and len(self.data.index) == len(other.data.index) and np.all(
                self.data.index == other.data.index)

    @classmethod
    def get(cls, ticker):
        return cls.stock_dict[ticker]

    @staticmethod
    def create_top_tickers_csv(filename="nasdaq_tickers.csv", number=30, sort_by="Market Cap", asc=False):
        tickers = pd.read_csv(open(filename, encoding="latin-1"))
        tickers = tickers.sort_values(by=sort_by, ascending=asc)
        tickers_by_ipo = tickers.loc[(pd.notnull(tickers['IPO Year'])) & (tickers['IPO Year'] <= 2006)]
        top_tickers = tickers_by_ipo.iloc[0:number, :]
        top_tickers.to_csv(f"tickers_{number}.csv")
        print(f"tickers_{number}.csv was created")

    @classmethod
    def get_stock_names(cls):
        ticker_lst = []
        for stock in cls.stock_list:
            ticker_lst.append(stock.name)
        return ticker_lst

    @classmethod
    def clear_stock_list(cls):
        cls.stock_list = []
        cls.stock_dict = {}

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
    def drop_stock(cls, name):
        idx = cls.get_stock_names().index(name)
        cls.stock_list.pop(idx)

    @classmethod
    def fetch_stock(cls, name):
        for stock in cls.stock_list:
            if stock.name == name:
                return stock
        raise KeyError('stock {name} not in stock_list')

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
        """calculates the strategies for every stock in stock_list
        every strategy should be single parameter so that:
            if [1,5,10] is given then the strategy is called three times with 1, then 5, then 10 as parameter
            the strategies are implemented so that a new column is created at each call (see sma_calc)

        arguments:
        strategy_dict: key is the name of the strategy, value is a list of parameters to calculate
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
    def strategy_looper_multi_param(cls, strategy_dict):
        """calculates the strategies for every stock in stock_list
        strategies should be multiple keyword argument

        arguments:
        strategy_dict: a dictionary where the key is the strategy name
                       and the value is another dictionary with arg:value pairs
        """

        for stock in cls.stock_list:
            for strategy, kwargs in strategy_dict.items():
                try:
                    strategy(stock, **kwargs)
                except ValueError as err:
                    print(err)
                    continue

    @classmethod
    def aggregate_data(cls):
        total_df = cls.stock_list[0].data
        for stock in cls.stock_list:
            total_df = pd.concat([total_df, stock.data])
        total_df.drop_duplicates(inplace=True)
        return total_df

    @classmethod
    def set_each(cls, total_df):
        """given a total_df it does the following for every stock in stock_list:
         set_data + lowercase + set_index + set_dates
        """
        stock_names = list(total_df['ticker'].unique())
        cls.clear_stock_list()
        cls.create_stock_list(stock_names)

        for stock in cls.stock_list:
            stock.set_data(total_df[total_df['ticker'] == stock.name].copy())
            stock.lowercase()
            stock.set_index()
            stock.set_dates()
            stock.set_sector()

    @staticmethod
    def generate_ranking(total_df, true_ranking=False, label_name='label_minmax_10'):
        """generates ranking for all stocks in stock_list for every day
        args:
        true_ranking: if True, calculates the ranking based on the true label and not the forecast
        label_name: one of 'label_minmax', label_max', 'label_avg', only used if true_ranking is True
        """

        if not true_ranking:
            if 'ranking' not in total_df.columns:
                total_df['ranking'] = np.nan

        if true_ranking:
            if 'true_ranking' not in total_df.columns:
                total_df['true_ranking'] = np.nan
            else:
                user_input = str(input('true_ranking already exists, want to overwrite: y/n'))
                if user_input.upper() in ['YES', 'Y']:
                    pass
                else:
                    raise InterruptedError

        for current_date in total_df.index.unique():
            if not true_ranking:
                if any(pd.isna(total_df.loc[total_df.index == current_date, 'forecast'])):
                    total_df.loc[current_date, 'ranking'] = np.nan
                else:
                    try:
                        total_df.loc[current_date, 'ranking'] = total_df.loc[current_date, 'forecast'].rank(
                            ascending=False)
                    except AttributeError:
                        total_df.loc[current_date, 'ranking'] = 1
            if true_ranking:
                if any(pd.isna(total_df.loc[total_df.index == current_date, label_name])):
                    total_df.loc[current_date, 'true_ranking'] = np.nan
                else:
                    try:
                        total_df.loc[current_date, 'true_ranking'] = total_df.loc[current_date, label_name].rank(
                            ascending=False)
                    except AttributeError:
                        total_df.loc[current_date, 'true_ranking'] = 1

    @staticmethod
    def ranking_to_dummy(total_df, threshold1=10, threshold2=20, threshold3=30, true_ranking=False):
        """turns ranking into dummy variables for all stocks in stock_list based on <= thresholds
           the new columns in stock.data are named as cat_1, cat_2, cat_3
        args:
        threshold: int,
        true_ranking: if True, generate dummy from true_ranking

        at the moment it is hard-coded to 3 different dummies (the 4th is left out):
            -being lower than threshold1
            -between threshold1 and 2
            -between 2 and 3
            -else
        e.g. 5, 10, 20 means that we differentiate top_5, top_5_to_10, top_10_to_20 and all other
        """
        if threshold3:
            num_stocks_per_day = Stock.get_stocks_per_date(total_df)['len_ticker'].min()
            if num_stocks_per_day <= threshold3:
                raise ValueError(f'thresholds do not differentiate {num_stocks_per_day} stocks')

        if 'ranking' not in total_df.columns:
            raise KeyError('ranking not available, use generate_ranking first')
        else:
            # df.between is inclusive by default on both sides, e.g. 0 <= total_df['ranking'] <= threshold1
            # df.mask changes the values of the column where the condition is True
            if threshold1:
                total_df['cat_1'] = np.where(total_df['ranking'].between(0, threshold1), 1, 0)
                total_df['cat_1'].mask(total_df['ranking'].isna(), np.nan, inplace=True)
            if threshold2:
                total_df['cat_2'] = np.where(total_df['ranking'].between(threshold1, threshold2), 1, 0)
                total_df['cat_2'].mask(total_df['ranking'].isna(), np.nan, inplace=True)
            if threshold3:
                total_df['cat_3'] = np.where(total_df['ranking'].between(threshold2, threshold3), 1, 0)
                total_df['cat_3'].mask(total_df['ranking'].isna(), np.nan, inplace=True)
        if true_ranking:
            if 'true_ranking' not in total_df.columns:
                raise KeyError('true_ranking not available, first use generate_ranking(true_ranking=True)')
            else:
                if threshold1:
                    total_df['true_cat_1'] = np.where(total_df['true_ranking'].between(0, threshold1), 1, 0)
                    total_df['true_cat_1'].mask(total_df['true_ranking'].isna(), np.nan, inplace=True)
                if threshold2:
                    total_df['true_cat_2'] = np.where(total_df['true_ranking'].between(threshold1, threshold2),
                                                        1, 0)
                    total_df['true_cat_2'].mask(total_df['true_ranking'].isna(), np.nan, inplace=True)
                if threshold3:
                    total_df['true_cat_3'] = np.where(total_df['true_ranking'].between(threshold2, threshold3),
                                                        1, 0)
                    total_df['true_cat_3'].mask(total_df['true_ranking'].isna(), np.nan, inplace=True)

    @staticmethod
    def get_stocks_per_date(total_df, verbose=False):
        if verbose:
            print('total_df', total_df)
        df1 = pd.DataFrame(total_df.groupby('date_')['ticker'].apply(lambda x: len(x)))
        df1.rename(columns={'ticker': 'len_ticker'}, inplace=True)
        df2 = pd.DataFrame(total_df.groupby('date_')['ticker'].apply(list))
        return pd.concat([df1, df2], axis=1)

    def yahoo_pull_data(self):
        yahoo_data = wb.DataReader(self.name, "yahoo", self.yahoo_pull_start_date, self.yahoo_pull_end_date)
        if not isinstance(yahoo_data, pd.DataFrame):
            raise TypeError('yahoo_data: not pandas.dataframe')
        yahoo_data['ticker'] = self.name
        self.set_data(yahoo_data)
        return self.data

    def drop_columns(self, cols=['high', 'low', 'open', 'adj_close']):
        """drops listed columns from self.data"""
        self.data.drop(labels=cols, axis=1, inplace=True, errors='ignore')

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
        if type(new_dataframe) is not pd.DataFrame:
            print(new_dataframe)
            raise TypeError('new_dataframe: not pandas.dataframe')
        self.data = new_dataframe

    def set_dates(self):
        if all([type(self.data.index[0]) is datetime.date, type(self.data.index[-1]) is datetime.date]):
            self.first_date = self.data.index[0]
            self.last_date = self.data.index[-1]
        else:
            print(f'type of first_date/last_date is {type(self.data.index[0])}, use set_index and repeat set_dates')
            self.first_date = self.data.index[0]
            self.last_date = self.data.index[-1]

    def set_sector(self):
        if self.data['sector'][0]:
            self.sector = self.data['sector'][0]
        else:
            self.sector = "Unknown"

    def set_yahoo_pull_start_date(self, new_date):
        self.yahoo_pull_start_date = new_date

    def set_yahoo_pull_end_date(self, new_date):
        self.yahoo_pull_end_date = new_date

    def calc_return(self, mode="log"):
        """calculates daily return"""
        tmp_df = pd.DataFrame({'close': self.data['close'], 'lag': self.data['close'].shift()})
        tmp_df.dropna(inplace=True)
        if mode == "normal":
            tmp_df['stock_return'] = (tmp_df['close'] / tmp_df['lag']) - 1
            self.data['stock_return'] = tmp_df['stock_return'].copy()
        elif mode == "log":
            tmp_df['log_return'] = np.log(tmp_df['close'] / tmp_df['lag'])
            self.data['log_return'] = tmp_df['stock_return'].copy()
        else:
            raise ValueError("mode not recognized in calc_return")

    def calc_variance(self, lookback=None, return_name='log_return'):
        """calculates variance based on daily returns series

        args:
        lookback: if None global variance is calc'd for each date,
                  if int then the global_var and a rolling.var() is calc'd for each date
        """
        if return_name not in self.data.columns:
            raise ValueError('invalid return_name in calc_variance')
        # !-!-!-!-! LABEL IS NOT THE ACTUAL LABEL FOR THAT DAY !-!-!-!-!
        # it is just the type of return we want to use
        # the true label will be self.data[label].shift(-1) when training
        # this is why we can calculate the rolling window without closed='left' since we already know the return for today

        # global variance calculation
        variance_lst = []
        for current_date in self.data.index:
            # ddof=1 to be consistent with the default degrees-of-freedom of pd.rolling.var
            vari = np.var(self.data.loc[:current_date, return_name].dropna(), ddof=1)
            variance_lst.append(vari)
        self.data['variance_global'] = variance_lst

        # here we have 'variance_global' for sure, so we can use it to replace the first nan entries created by rolling
        if lookback:
            self.data[f'variance_{lookback}'] = self.data[return_name].rolling(lookback).var()
            self.data[f'variance_{lookback}'].mask(self.data[f'variance_{lookback}'].isna(),
                                                   self.data['variance_global'], inplace=True)

    def sma_calc(self, period=20, only_last_date=False):
        """calculates simple moving average with window length=period
        creates a new column for the sma_{period} in self.data
        if data is too short for given period a ValueError is raised
        """
        if only_last_date:
            if f'sma_{period}' not in self.data.columns:
                self.data[f'sma_{period}'] = np.nan
                self.data.loc[self.last_date, f'sma_{period}'] = \
                    self.data['close'][-period:].mean()
        else:
            if len(self.data.index) <= period:
                raise ValueError(
                    f'len(self.data.index)={len(self.data.index)} for {self.name} is shorter than period={period}')
            if f'sma_{period}' not in self.data.columns:
                self.data[f'sma_{period}'] = self.data['close'].rolling(window=period).mean()
            else:
                user_input = str(input(f'sma_{period} already exists, want to recalculate: y/n'))
                if user_input.upper() in ['YES', 'Y']:
                    self.data[f'sma_{period}'] = self.data['close'].rolling(window=period).mean()
                else:
                    raise InterruptedError

    def sma_cross(self, sma_short_days, sma_long_days):
        if not f'sma_{sma_long_days}' in self.data.columns:
            self.sma_calc(period=sma_long_days)
        if not f'sma_{sma_short_days}' in self.data.columns:
            self.sma_calc(period=sma_short_days)

        below_or_above = np.where(self.data[f'sma_{sma_short_days}'] < self.data[f'sma_{sma_long_days}'], 0,
                                  np.where(self.data[f'sma_{sma_short_days}'] > self.data[f'sma_{sma_long_days}'], 1,
                                           np.nan))

        crosses = [np.nan]
        for i in range(len(below_or_above) - 1):
            crosses.append(below_or_above[i + 1] - below_or_above[i])

        self.data[f'sma_cross_{sma_short_days}_{sma_long_days}'] = crosses

    def sma_to_input(self, only_last_date=False):
        """turns sma levels into input for one stock used for the forecast model
           calculates the distance: self.data['sma_..._distance'] = self.data['close'] - self.data['sma_...']
           for all columns starting with sma (sma_9, sma_14, sma_100) except sma_cross !!!

        args:
        only_last_date: if True then self.data['close'] - self.data['sma_...'] is only calc'd for self.last_date,
                        if False then all of self.data['close'] - self.data['sma_...'] is calc'd
        """
        sma_cols = pd.Series([col for col in self.data if col.startswith('sma_')])
        if not sma_cols.empty:
            sma_cols.drop(sma_cols.index[sma_cols.str.contains('cross')], inplace=True)
            sma_cols.drop(sma_cols.index[sma_cols.str.endswith('distance')], inplace=True)
        else:
            print(f'No sma column: {self.name}')
            return 0

        if only_last_date:
            current_price = self.get_price(as_of=self.last_date)
            # percentage distance from current price
            for col in sma_cols:
                if f'{col}_distance' not in self.data.columns:
                    self.data[f'{col}_distance'] = np.nan
                self.data.loc[self.last_date, f'{col}_distance'] = \
                    (current_price - self.data.loc[self.last_date, col]) / current_price
        else:
            # percentage distance from close price
            for col in sma_cols:
                self.data[f'{col}_distance'] = \
                    (self.data['close'] - self.data[col]) / self.data['close']

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
        """
        arguments:
        from_date: datetime.date object determining the left edge of the plot plus the lookback of calculating tech levels
        to_date: similar to from_date only the other end (most of the time this is set to today)
        show_tech_levels: if True technical levels are calculated inside the window provided by from_date-to_date
        kwargs:
        1)tech_width -- own argument determining the width of a tech_level
        2)other kwargs passed to scipy.signal.find_peaks"""
        if type(from_date) is not datetime.date:
            try:
                from_date = datetime.date(from_date)
            except TypeError:
                print(f'type of from_date: {type(from_date)}')
        if type(to_date) is not datetime.date:
            try:
                to_date = datetime.date(to_date)
            except TypeError:
                print(f'type of to_date: {type(from_date)}')
        chunk = self.data.loc[from_date: to_date]
        first_date = chunk.first_valid_index()
        last_date = chunk.last_valid_index()
        plt.plot(chunk['close'])
        plt.title(self.name)

        if show_tech_levels:
            tech_levels = self.get_tech_levels(from_date=first_date, to_date=last_date, **kwargs)
            # y_coords is the average of the min and max of that tech_level -- maybe plot rectangle and not avg later
            # sublist[0] is a list with 2 values min, max
            y_coords = [(sum(sublist[0]) / len(sublist[0])) for sublist in tech_levels]
            red_color = [[1, 0, 0]] * len(tech_levels)
            # sublist[1] is a number representing the length of the sublist tech level
            # alphas = [min((sublist[1]/10), 1) for sublist in tech_levels]
            alphas = [sublist[1] for sublist in tech_levels]
            medium = [i for i in range(len(alphas)) if alphas[i] == 0.5]
            for elem in medium:
                red_color[elem] = [0, 0, 1]
            tmp = list(zip(red_color, alphas))
            red_with_alphas = [tuning.flatten(elem, num_iter=1) for elem in tmp]
            plt.hlines(y_coords, xmin=first_date, xmax=last_date, colors=red_with_alphas)
        plt.show()

    def get_price(self, as_of):
        try:
            return self.data.loc[as_of, 'close']
        except KeyError:
            if self.last_date < as_of:
                return self.data.loc[self.last_date, 'close']
            else:
                if as_of.weekday() == 6:
                    previous_day = as_of - datetime.timedelta(days=2)
                else:
                    previous_day = as_of - datetime.timedelta(days=1)
                price = self.get_price(previous_day)
                return price
        except TypeError:
            print(f'Stock.data is not set for {self.name}. First fill it from yahoo or sql.')
            return 0


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
        kwargs: passed to scipy.signal.find_peaks():
                height because the volume series is centered to 0 """

        try:
            height = kwargs['height']
        except KeyError:
            height = 0
        # prepare data chunk
        data_chunk = self.get_price_range(from_date, to_date)
        auto_size = min(10, len(data_chunk['volume']))
        max_dates_idx = pd.Series(data_chunk['volume'].sort_values()[-auto_size:].index)
        # convert to stationary
        stationary_volume = tuning.stationary_maker(data_chunk['volume'])
        # center
        centered_volume = stationary_volume - stationary_volume.mean()
        # height of peaks in standard deviation units
        st_dev = centered_volume.std()
        height_in_st_dev = height * st_dev
        kwargs['height'] = height_in_st_dev

        # find peaks
        peak_vol_idx, _ = find_peaks(centered_volume, **kwargs)
        peak_vol_dates = data_chunk['date_'][peak_vol_idx]
        # unique_dates = peak_vol_dates.append(max_dates_idx, ignore_index=True)
        unique_dates = pd.concat([peak_vol_dates, max_dates_idx], ignore_index=True)
        peaks_df = data_chunk.loc[self.data['date_'][unique_dates.drop_duplicates()], ('date_', 'volume')]
        return peaks_df

    def get_tech_levels(self, from_date=datetime.date(2000, 1, 1), to_date=datetime.date.today(), **kwargs):
        """kwargs:
        1)tech_width -- own argument determining the width of a level
        2)consider_volume = True
        3) volume_height -- used as the height param for find_peaks in get_high_volumes, suggested range: 0,2
        4)other kwargs passed to scipy.signal.find_peaks
            """
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
            volume_height = kwargs['volume_height']
            kwargs.pop('volume_height')
        except KeyError:
            volume_height = 0
        try:
            volume_prominence = kwargs['volume_prominence']
            kwargs.pop('volume_prominence')
        except KeyError:
            volume_prominence = 0
        try:
            kwargs['rel_height']
        except KeyError:
            kwargs['rel_height'] = 0.5

        data_chunk = self.get_price_range(from_date, to_date)
        scaled = data_chunk['close'] / data_chunk['close'][-1]
        peaks, _ = find_peaks(scaled, **kwargs)
        troughs, _ = find_peaks(scaled * (-1), **kwargs)

        if consider_volume:
            high_vol_df = self.get_high_volumes(from_date, to_date, **{'height': volume_height,
                                                                       'prominence': volume_prominence})

        high_price_df = data_chunk.iloc[
                        np.concatenate((peaks, troughs)), :]
        high_price_df = high_price_df.loc[:, ('date_', 'close')]

        if consider_volume:
            unique_dates = pd.concat([high_vol_df['date_'], high_price_df['date_']], ignore_index=True)
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
                    tech_levels[i].extend(list(tech_df.iloc[[i, k], close_column_idx]))

        # dropping empty sublists of tech_levels list-of-lists
        tech_levels = [elem for elem in tech_levels if elem]

        # tech_levels[i][0] -- min-max y_coord
        # tech_levels[i][1] -- how many peaks were in that tech_level -- stregth or alpha for plotting
        # rounding is done so that 0,1: 0 -- 2,3: 0.5 -- 4,5,6,7,8,9,10,...: 1
        tech_levels = [[[min(sublist), max(sublist)],
                        0 if len(sublist) in [0, 1] else 0.5 if len(sublist) in [2] else 1] for sublist in tech_levels]
        return tech_levels

    def tech_levels_to_input(self, current_date, lookback, **kwargs):
        """returns the distance from the closest strong and medium tech levels for one date
        for the looper version see tech_level_input_calc

        tech levels are calculated based on a [current_date - lookback, current_date] window
        kwargs: passed to get_tech_levels
        """
        fr = current_date - datetime.timedelta(days=lookback)
        tech_levels = self.get_tech_levels(from_date=fr, to_date=current_date, **kwargs)
        # strong_list and medium_list contain tuples of min-max values for each tech_level
        # to check if current_price between min and max for one tech_level convert to range
        strong_list = [tech_level[0] for tech_level in tech_levels if tech_level[1] == 1]
        medium_list = [tech_level[0] for tech_level in tech_levels if tech_level[1] == 0.5]
        current_price = self.get_price(as_of=current_date)

        # positive distance: current price greater than tech level
        # negative distance: current price smaller than tech level
        # zero distance: within a tech level
        strong_distance = [0 for s in strong_list if current_price in range(round(s[0]), round(s[1]))]
        medium_distance = [0 for m in medium_list if current_price in range(round(m[0]), round(m[1]))]

        if strong_distance:
            strong_distance_percent = 0
        else:
            try:
                strong_distance = current_price - tuning.closest_number(current_price, strong_list)
                strong_distance_percent = strong_distance / current_price
            except TypeError:
                if tuning.closest_number(current_price, strong_list) is None:
                    strong_distance_percent = None

        if medium_distance:
            medium_distance_percent = 0
        else:
            try:
                medium_distance = current_price - tuning.closest_number(current_price, medium_list)
                medium_distance_percent = medium_distance / current_price
            except TypeError:
                if tuning.closest_number(current_price, medium_list) is None:
                    medium_distance_percent = None

        return strong_distance_percent, medium_distance_percent

    def tech_level_input_calc(self, lookback, init_size=50, verbose=False, **kwargs):
        """calculates everything related to tech_levels for one stock for the whole training data with daily iteration
        
        calls tech_levels_to_input within a daily loop looking only backwards which calls get_tech_levels
        generates the x variable for every day that can be used in the forecast model
        
        
        arguments:
        lookback: window size for calculating the technical levels, passed to tech_levels_to_input and get_tech_levels
        init_size: the size of the data chunk below which no technical levels are calculated, default = 50
        verbose: print stuff
        """
        for row in range(init_size, len(self.data.index)):
            current_date = self.data.index[row]
            if verbose:
                print('current_date', current_date)
                print('before', self.data.loc[current_date, ['tech_strong', 'tech_medium']])
            # unpack tech_levels_to_input and fill self.data['tech_strong', 'tech_medium']
            self.data.loc[current_date, 'tech_strong'], \
            self.data.loc[current_date, 'tech_medium'] \
                = self.tech_levels_to_input(current_date, lookback, **kwargs)
            if verbose:
                print('after', self.data.loc[current_date, ['tech_strong', 'tech_medium']])
                print(" - - - - - - - - - - - ")

    def generate_simple_forecast(self, label_str, method, skip_cols, init_size=200, verbose=False, **kwargs):
        """Generates forecast for the next day with fitting the regressor to a growing lice of the data
        This can only be used for elementary/simple/linear forecasts that will be combined with boosting later

        label_str: the name of the label as a string
        method: one function from forecast.py
        init_size: the first prediction is estimated for init_size + 1
        skip_cols: column names not to use in fitting as a string or list of strings
        **kwargs: passed to the selected method

        stock_return can be used as a covariate because it is known at the time of the forecast
        label_stock_return is tomorrow's stock_return
        so for a specific date: regress label_stock_return on stock_return, lag_stock_return, lag2_stock_return ...

        """
        for row in range(init_size, len(self.data.index)):
            current_date = self.data.index[row]

            # relevant_data only contains the X,y pairs needed for fitting and forecasting
            relevant_data = self.data.drop(skip_cols, axis=1, errors='ignore')
            if verbose:
                print('columns used for fitting', relevant_data.columns)
            training_data = relevant_data.loc[:current_date, :].copy()

            # fitting the model
            model = method(training_data, label_str, **kwargs)
            # forecast
            # new observation for current_date+1 without the label y
            try:
                forecast_date = self.data.index[row + 1]
                new_observation_X = relevant_data.loc[forecast_date, relevant_data.columns != label_str]
                # reshape so that one sample is represented as a 2D array
                new_observation_X = np.array(new_observation_X)
                new_observation_X = new_observation_X.reshape(1, -1)

                prediction = model.predict(new_observation_X)
                self.data.loc[forecast_date, 'forecast'] = prediction
            except IndexError:
                return model

    @classmethod
    def cross_validate(cls, total_df, label_str, skip_cols, method, init_ratio=0.99, **kwargs):
        """do not include the label in skip_cols"""

        if 'forecast' not in total_df.columns:
            total_df['forecast'] = np.nan

        if 'forecast' not in skip_cols:
            skip_cols.append('forecast')

        if not 0 < init_ratio < 1:
            raise ValueError("init_ratio should be between 0 and 1")
        init_size = round(len(total_df.index) * init_ratio)
        init_date = total_df.index[init_size]
        size = len(total_df.loc[:init_date, :])

        for current_date in total_df['date_'].loc[init_date:].unique():
            print('current_date', current_date)
            # forecast date is the second element (1st index) of the unique upcoming days
            try:
                forecast_date = total_df['date_'].loc[current_date:].unique()[1]
            except IndexError:
                print('Done')
                return model

            # dropping columns not needed for forecasting
            relevant_df = total_df.drop(skip_cols, axis=1, errors='ignore').copy()

            # gradient_booster cannot handle NaNs but has feature_importance
            if method.__name__ == 'gradient_booster':
                categorical_df = relevant_df.select_dtypes(include="category")
                for col_name in categorical_df.columns:
                    categorical_df[col_name] = categorical_df[col_name].astype('object')
                relevant_df.drop(relevant_df.select_dtypes(include="category"), axis=1, inplace=True)
                relevant_df.fillna(100000, inplace=True)

            # training data until the current_date in the loop
            training_slice = relevant_df.loc[:current_date, :].copy()
            # fitting
            model = method(training_slice, label_str, **kwargs)

            # forecasting
            new_observation_X = relevant_df.loc[forecast_date, relevant_df.columns != label_str].copy()
            if len(new_observation_X.index) == 1:
                new_observation_X = np.array(new_observation_X).reshape(1, -1)
            prediction = model.predict(new_observation_X)
            total_df.loc[forecast_date, 'forecast'] = prediction
            print(total_df.loc[forecast_date, 'forecast'])
            # reshape so that one sample is represented as a 2D array
            # Reshape your data either using array.reshape(-1, 1) if your data has a single feature
            # or array.reshape(1, -1) if it contains a single sample.

    @classmethod
    def fit_predict(cls, total_df, label_str, predict_col_name, skip_cols, method, train_ratio=0.5, predict="train", **kwargs):
        """do not include the label in skip_cols"""

        if predict_col_name not in total_df.columns:
            total_df[predict_col_name] = np.nan

        if predict_col_name not in skip_cols:
            skip_cols.append(predict_col_name)

        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio should be between 0 and 1")
        init_size = round(len(total_df.index) * train_ratio)
        init_date = total_df.index[init_size]
        print('init_date', init_date)

        # dropping columns not needed for forecasting
        relevant_df = total_df.drop(skip_cols, axis=1, errors='ignore').copy()

        # gradient_booster cannot handle NaNs but has feature_importance
        if method.__name__ == 'gradient_booster':
            categorical_df = relevant_df.select_dtypes(include="category")
            for col_name in categorical_df.columns:
                categorical_df[col_name] = categorical_df[col_name].astype('object')
            relevant_df.drop(relevant_df.select_dtypes(include="category"), axis=1, inplace=True)
            relevant_df.fillna(100000, inplace=True)

        training_slice = relevant_df.loc[:init_date, :].copy()
        print('last date of training_slice', training_slice.iloc[-1, :])
        # fitting
        model = method(training_slice, label_str, **kwargs)
        print('unique_dates', total_df['date_'].loc[init_date:].unique())
        for current_date in total_df['date_'].loc[init_date:].unique():
            print('current_date', current_date)
            # forecast date is the second element (1st index) of the unique upcoming days
            try:
                forecast_date = current_date
            except IndexError:
                print('Done')
                return model

            # forecasting
            new_observation_X = relevant_df.loc[forecast_date, relevant_df.columns != label_str].copy()
            if len(new_observation_X.index) == 1:
                new_observation_X = np.array(new_observation_X).reshape(1, -1)
            prediction = model.predict(new_observation_X)
            total_df.loc[forecast_date, predict_col_name] = prediction
            # print(total_df.loc[forecast_date, predict_col_name])



    @staticmethod
    def train_test_split(total_df, train_ratio=0.5, verbose=True):
        """returns a train df and a test df based on a train_ratio split considering the time series structure"""
        total_df.sort_index(inplace=True)
        max_idx = round(len(total_df.index) * train_ratio)
        last_date = total_df.index[max_idx]
        first_test_date = total_df.index.unique()[list(total_df.index.unique()).index(last_date)+1]
        if verbose:
            print(f"training data: {total_df.index[0]} -- {last_date}")
            print(f"test data: {first_test_date} -- {total_df.index[-1]}")

        train = total_df.loc[:last_date, :].copy()
        test = total_df.loc[first_test_date:, :].copy()
        return train, test

    @staticmethod
    def fit_AR(train_df, label_str, p, const=True, group_by="sector", predict=True, verbose=True):
        """fits an AR(p) model separately for every group created by group_by"""
        train_df.sort_index(inplace=True)

        X_cols = []
        models = {}
        # label_str is either label_stock_return or label_log_return -- get rid of label_ and put _lag{i} to the end
        lag_name = label_str.split("_")
        lag_name.pop(0)
        lag_name_str = "_".join(lag_name)

        for i in range(p):
            X_cols.append(lag_name_str + "_lag" + str(i))
        if verbose:
            if const:
                print(f"regressing: {label_str} ~ const + {X_cols}")
                print(f"group by: {group_by}")
            else:
                print(f"regressing: {label_str} ~ {X_cols}")
                print(f"group by: {group_by}")

        ols_columns = X_cols + [label_str]

        df_collector = []
        if group_by == "sector":
            for sector in train_df['sector_encoded'].unique():
                df = train_df.loc[train_df['sector_encoded'] == sector, ols_columns]
                df.dropna(axis=0, subset=ols_columns, inplace=True)
                df_sector_all_columns = train_df.loc[train_df['sector_encoded'] == sector, :]
                df_sector_all_columns.dropna(axis=0, subset=ols_columns, inplace=True)

                Y = df.loc[:, label_str]
                X = df.loc[:, X_cols]
                if const:
                    X = sm.add_constant(X)

                model = sm.OLS(Y, X).fit()
                models[sector] = model

                if predict:
                    # predict using training data
                    predictions = model.predict()
                    df_sector_all_columns[f"AR_{p}"] = predictions
                    df_collector.append(df_sector_all_columns)
            if verbose:
                print(f"created column: AR_{p}")

        else:
            raise ValueError("group_by can only be 'sector'")

        merged = pd.DataFrame()
        for frame in df_collector:
            merged = pd.concat([merged, frame], axis=0)
        merged.sort_index(inplace=True)
        return merged, models

    @staticmethod
    def predict_AR(test_df, models, group_by="sector", verbose=True):
        # all models have the same lag lenght p, so we only need one
        selected_key = list(models.keys())[0]
        params = list(models[selected_key].params.index)
        if "const" in params:
            const = True
            params.pop(params.index("const"))
        else:
            const = False

        X_cols = params.copy()
        # last letter of the largest lag + 1 (since that starts from 0)
        try:
            p = int(X_cols[-1][-1]) + 1
        except ValueError:
            p = str(input("What is the lag order of this model? p = ..."))

        if verbose:
            print(f"Lag length: {p}")
            print(f"Covariates: {'const' if const else ''} + {X_cols}")
        df_collector = []
        if group_by == "sector":
            for sector, model in models.items():
                test_df_by_sector = test_df.loc[test_df['sector_encoded'] == sector, :].copy()
                X = test_df.loc[test_df['sector_encoded'] == sector, X_cols].copy()
                if const:
                    X = sm.add_constant(X)
                prediction = model.predict(exog=X)
                test_df_by_sector[f'AR_{p}'] = prediction
                df_collector.append(test_df_by_sector)

        merged = pd.DataFrame()
        for frame in df_collector:
            merged = pd.concat([merged, frame], axis=0)
        merged.sort_index(inplace=True)
        return merged





