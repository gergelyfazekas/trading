import concurrent.futures
import datetime
# turn of FutureWarning
import math
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas_datareader import data as wb
import pandas as pd
import numpy as np
import database
import sys
import matplotlib.pyplot as plt
import stock_class
from stock_class import Stock

# turn off chained assignment warning
pd.options.mode.chained_assignment = None


class Portfolio:
    portfolio_list = []

    def __init__(self, genome, cash):
        if type(cash) in (int, np.integer, float, np.float16, np.float32, np.float64):
            __class__.portfolio_list.append(self)
            self.exchange = "NASDAQ"
            self.genome = genome
            self.cash_init = cash
            self.cash_current = cash
            self.cash_spent = 0
            self.number_of_stocks = int
            self.currencies = list
            self.log = pd.DataFrame({'date_': [np.nan] * stock_class.PLACEHOLDER,
                                     'stock_name': [np.nan] * stock_class.PLACEHOLDER,
                                     'direction': [np.nan] * stock_class.PLACEHOLDER,
                                     'amount': [np.nan] * stock_class.PLACEHOLDER,
                                     'price': [np.nan] * stock_class.PLACEHOLDER,
                                     'value': [np.nan] * stock_class.PLACEHOLDER})
            self.balance = pd.DataFrame({'stock_name': [np.nan] * stock_class.PLACEHOLDER,
                                         'amount': [np.nan] * stock_class.PLACEHOLDER,
                                         'price': [np.nan] * stock_class.PLACEHOLDER,
                                         'value': [np.nan] * stock_class.PLACEHOLDER,
                                         'sector': [np.nan] * stock_class.PLACEHOLDER})
            self.total_portfolio_value_hist = []
            self.portfolio_return_hist = []

    def __eq__(self, other):
        return self.genome == other.genome and self.cash_init == other.cash_init and self.exchange == other.exchange

    @property
    def total_portfolio_value(self):
        if self.balance.last_valid_index() is not None:
            return self.cash_current + self.balance['value'].sum()
        else:
            return self.cash_current

    @property
    def entropy_stock(self):
        # if all self.balance['value'] is np.nan --> weights would be np.nan --> fitness would be np.nan
        if not self.balance['value'].first_valid_index():
            return 0
        else:
            cutoff = 0.000001
            total_value = self.balance['value'].sum()
            weights = np.array(self.balance['value'] / total_value)
            weights[weights < cutoff] = cutoff
            return -1 * np.sum(weights * np.log(weights))

    @property
    def entropy_sector(self):
        if not self.balance['value'].first_valid_index():
            return 0
        else:
            cutoff = 0.000001
            sector_values = list(self.balance.groupby(by="sector")['value'].sum())
            total_value = self.balance['value'].sum()
            weights = np.array(sector_values / total_value)
            weights[weights < cutoff] = cutoff
            return -1 * np.sum(weights * np.log(weights))

    @property
    def proportion_invested(self):
        return self.cash_spent / self.cash_init

    @property
    def sectors(self):
        return list(self.balance['sector'].unique())

    def got_enough_cash(self, value):
        if type(value) in (int, np.integer, float, np.float16, np.float32, np.float64):
            if value >= 0:
                if value <= self.cash_current:
                    return True
                else:
                    # print(f'Current cash ({self.cash_current}) not enough for value ({value})')
                    return False
            else:
                # print(f'Negative value: {value}')
                return False
        else:
            raise TypeError('value not in (int, float)')

    def got_enough_amount(self, stock, amount):
        if type(amount) in (int, np.integer, float, np.float16, np.float32, np.float64):
            if amount <= 0:
                if abs(amount) <= any(self.balance.loc[self.balance['stock_name'] == stock.name, 'amount']):
                    return True
                else:
                    # print(f'Balance contains less amount of {stock.name}'
                    #       f' ({self.balance.loc[self.balance["stock_name"] == stock.name, "amount"]})'
                    #       f' than the required ({abs(amount)})')
                    return False
            else:
                # print(f'Positive amount: {amount}')
                # print('here1')
                return False
        elif type(amount) is pd.Series:
            if any(amount <= 0):
                if any(abs(amount)) <= any(self.balance.loc[self.balance['stock_name'] == stock.name, 'amount']):
                    return True
                else:
                    # print(f'Balance contains less amount of {stock.name}'
                    #       f' ({self.balance.loc[self.balance["stock_name"] == stock.name, "amount"]})'
                    #       f' than the required ({abs(amount)})')
                    return False
            else:
                # print(f'Positive amount: {amount}')
                # print('here2')
                return False
        else:
            raise TypeError('value not in (int, float)')

    def get_stock_amount(self, stock):
        if self.balance.loc[self.balance['stock_name'] == stock.name, 'amount'].empty:
            return 0
        else:
            return self.balance.loc[self.balance['stock_name'] == stock.name, 'amount']

    def deduct_cash(self, amount):
        if type(amount) in (int, np.integer, float, np.float16, np.float32, np.float64):
            if amount >= 0:
                self.cash_current -= amount
                return 0
        else:
            amount = float(amount)
            self.deduct_cash(amount)

    def add_cash(self, amount):
        if type(amount) in (int, np.integer, float, np.float16, np.float32, np.float64):
            if amount >= 0:
                self.cash_current += amount
                return 0
        else:
            amount = float(amount)
            self.add_cash(amount)

    def update_cash_spent(self, value):
        """accepts negative value as well meaning that we get back cash by selling a stock"""
        try:
            self.cash_spent += float(value)
        except TypeError:
            print("cash_spent not updated with value, ", value)

    def update_balance_transaction(self, stock, amount, price, value, sector):
        non_zero = True
        try:
            if np.round(abs(amount), 2) == 0:
                non_zero = False
        except ValueError:
            if any(np.round(abs(amount), 2) == 0):
                non_zero = False
        if non_zero:
            # if stock is in the list already
            if stock.name in self.balance['stock_name'].unique():
                # update
                self.balance.loc[self.balance['stock_name'] == stock.name, 'amount'] += amount
                self.balance.loc[self.balance['stock_name'] == stock.name, 'price'] = price
                self.balance.loc[self.balance['stock_name'] == stock.name, 'value'] = \
                    price * self.balance.loc[self.balance['stock_name'] == stock.name, 'amount']
            else:
                # if stock is not in the list yet put it in the last row
                last_idx = self.balance.loc[:, 'stock_name'].last_valid_index()
                if type(last_idx) in (int, np.integer, float, np.float16, np.float32, np.float64):
                    self.balance.iloc[last_idx + 1, self.balance.columns.get_loc('stock_name')] = stock.name
                    self.balance.iloc[last_idx + 1, self.balance.columns.get_loc('amount')] = amount
                    self.balance.iloc[last_idx + 1, self.balance.columns.get_loc('price')] = price
                    self.balance.iloc[last_idx + 1, self.balance.columns.get_loc('value')] = value
                    self.balance.iloc[last_idx + 1, self.balance.columns.get_loc('sector')] = sector

                else:
                    # if idx is None then put it in the first row
                    self.balance.iloc[0, self.balance.columns.get_loc('stock_name')] = stock.name
                    self.balance.iloc[0, self.balance.columns.get_loc('amount')] = amount
                    self.balance.iloc[0, self.balance.columns.get_loc('price')] = price
                    self.balance.iloc[0, self.balance.columns.get_loc('value')] = value
                    self.balance.iloc[0, self.balance.columns.get_loc('sector')] = sector

    def update_log(self, as_of, stock, direction, amount, price, value):
        non_zero = True
        try:
            if np.round(abs(amount), 2) == 0:
                non_zero = False
        except ValueError:
            if any(np.round(abs(amount), 2) == 0):
                non_zero = False
        if non_zero:
            idx = self.log['stock_name'].last_valid_index()
            if type(idx) in (int, np.integer, float, np.float16, np.float32, np.float64):
                try:
                    self.log.iloc[idx + 1, self.log.columns.get_loc('date_')] = as_of
                    self.log.iloc[idx + 1, self.log.columns.get_loc('stock_name')] = stock.name
                    self.log.iloc[idx + 1, self.log.columns.get_loc('direction')] = direction
                    self.log.iloc[idx + 1, self.log.columns.get_loc('amount')] = amount
                    self.log.iloc[idx + 1, self.log.columns.get_loc('price')] = price
                    self.log.iloc[idx + 1, self.log.columns.get_loc('value')] = value
                except IndexError:
                    print('max index reached, iloc cannot expand the dataframe', self.log)

            elif idx is None:
                self.log.iloc[0, self.log.columns.get_loc('date_')] = as_of
                self.log.iloc[0, self.log.columns.get_loc('stock_name')] = stock.name
                self.log.iloc[0, self.log.columns.get_loc('direction')] = direction
                self.log.iloc[0, self.log.columns.get_loc('amount')] = amount
                self.log.iloc[0, self.log.columns.get_loc('price')] = price
                self.log.iloc[0, self.log.columns.get_loc('value')] = value
            else:
                raise NotImplementedError

    def update_balance_eod(self, as_of):
        if self.balance.last_valid_index():
            for stock_name in self.balance['stock_name']:
                if type(stock_name) is str:
                    stock = Stock.get(stock_name)
                    price = stock.get_price(as_of)
                    # update price
                    self.balance.loc[self.balance['stock_name'] == stock_name, "price"] = price
                    # update value = price * amount
                    self.balance.loc[self.balance['stock_name'] == stock_name, "value"] = \
                        price * self.balance.loc[self.balance['stock_name'] == stock_name, "amount"]
        self.update_total_portfolio_value_hist(as_of)
        self.update_portfolio_return_hist(as_of)

    def update_number_of_stocks(self):
        self.number_of_stocks = len(self.balance['stock_name'].unique())

    def buy(self, stock, amount, as_of):
        # if not type(stock) is Stock:
        #     raise TypeError(f'Not Stock instance! type given: {type(stock)}')
        if type(as_of) is not datetime.date:
            raise TypeError(f'as_of should be datetime.date and not {type(as_of)}')

        if amount >= 0:
            price = stock.get_price(as_of)
            value = price * amount

            if self.got_enough_cash(value):
                self.deduct_cash(value)
                self.update_cash_spent(value)
                self.update_balance_transaction(stock, amount, price, value, stock.sector)
                self.update_log(as_of=as_of, stock=stock, direction='buy', amount=amount, price=price, value=value)

        else:
            raise ValueError('buy requires a positive amount')

    def sell(self, stock, amount, as_of):
        if type(stock) is not Stock:
            raise TypeError(f'Not Stock instance! type given: {type(stock)}')

        if type(amount) is pd.Series:
            if any(amount <= 0):
                sell_price = stock.get_price(as_of)
                sell_value = sell_price * amount
                # if the value is too small then just exit
                try:
                    if round(sell_value, 3) == 0:
                        return 0
                except ValueError:
                    if any(round(sell_value, 3) == 0):
                        return 0
            else:
                return 0
        elif type(amount) in (int, np.integer, float, np.float16, np.float32, np.float64):
            if amount <= 0:
                sell_price = stock.get_price(as_of)
                sell_value = sell_price * amount
                # if the value is too small then just exit
                try:
                    if round(sell_value, 3) == 0:
                        return 0
                except ValueError:
                    if any(round(sell_value, 3) == 0):
                        return 0

            else:
                return 0
        else:
            raise TypeError("amount not numeric")

        # amount is negative, value is negative, price is positive
        if self.got_enough_amount(stock, amount):
            self.add_cash(abs(sell_value))
            self.update_cash_spent(sell_value)
            self.update_balance_transaction(stock, amount, sell_price, sell_value, stock.sector)
            self.update_log(as_of=as_of, stock=stock, direction='sell', amount=amount, price=sell_price,
                            value=sell_value)
        else:
            # print('Do not have enough amount to sell')
            pass

    def calc_variance(self, lookback=None):
        """calculates total portfolio variance based on daily portfolio returns

        args:
        lookback: if None global variance is calc'd for each date, e.g. var(data[:current_date])
                  if int then the global_var and a rolling.var() is calc'd for each date
        """
        # check if returns exist
        if 'stock_return' not in self.data.columns:
            user_input = str(input('stock_return does not exist, want to calculate: y/n'))
            if user_input.upper() in ['YES', 'Y']:
                self.calc_return()
            else:
                raise InterruptedError('use calc_return before calc_variance')

        # check if variance_global exists
        if 'variance_global' in self.data.columns:
            user_input = str(input('variance_global already exists, want to recalculate: y/n'))
        else:
            user_input = "Y"

        # if not in it or user wants to recalculate
        if 'variance_global' not in self.data.columns or user_input.upper() in ['YES', 'Y']:
            variance_lst = []
            for current_date in self.data.index:
                # ddof=1 to be consistent with the default degrees-of-freedom of pd.rolling.var
                vari = np.var(self.data.loc[:current_date, 'stock_return'].dropna(), ddof=1)
                variance_lst.append(vari)
            self.data['variance_global'] = variance_lst

        # here we have 'variance_global' for sure, so we can use it to replace the first nan entries created by rolling
        if lookback:
            if f'variance_{lookback}' in self.data.columns:
                user_input = str(input(f'variance_{lookback} already exists, want to recalculate: y/n'))

            if f'variance_{lookback}' not in self.data.columns or user_input.upper() in ['YES', 'Y']:
                self.data[f'variance_{lookback}'] = self.data['stock_return'].rolling(lookback).var()
                self.data[f'variance_{lookback}'].mask(self.data[f'variance_{lookback}'].isna(),
                                                       self.data['variance_global'], inplace=True)

    def update_total_portfolio_value_hist(self, current_date):
        """updates the list of historical portfolio values, needs to be updated every EoD
        update_balance_eod triggers / calls this function
        """
        self.total_portfolio_value_hist.append((current_date, self.total_portfolio_value))

    def update_portfolio_return_hist(self, current_date):
        """updates the list of historical portfolio returns, needs to be updated every EoD
        update_balance_eod triggers / calls this function

        the return is calc'd from the history of total_portfolio_values
        total_portfolio_value is calc'd from cash_current + balance
        """
        if self.total_portfolio_value_hist:

            portfolio_value_today, today_idx = \
                [(item[1], self.total_portfolio_value_hist.index(item)) for item in self.total_portfolio_value_hist \
                 if item[0] == current_date][0]

            if len(self.total_portfolio_value_hist) >= 2:
                previous_idx = today_idx - 1
                portfolio_value_yesterday = self.total_portfolio_value_hist[previous_idx][1]
                self.portfolio_return_hist.append((current_date, portfolio_value_today / portfolio_value_yesterday))
            else:
                self.portfolio_return_hist.append((current_date, 1))
        else:
            self.portfolio_return_hist.append((current_date, 1))
