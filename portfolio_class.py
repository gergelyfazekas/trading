import concurrent.futures
import datetime
# turn of FutureWarning
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
DATE = datetime.date(2007, 1, 12)


class Portfolio:
    portfolio_list = []

    def __init__(self, genome, cash):
        if isinstance(cash, (int, float)):
            __class__.portfolio_list.append(self)
            self.exchange = "NASDAQ"
            self.genome = genome
            self.cash_init = cash
            self.cash_current = cash
            self.cash_spent = 0
            self.number_of_stocks = int
            self.sectors = list
            self.currencies = list
            self.log = pd.DataFrame({'date': [np.nan] * stock_class.PLACEHOLDER,
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
            self.total_portfolio_value = self.cash_init
            self.proportion_invested = self.cash_spent / self.cash_init

    def __eq__(self, other):
        if self.genome == other.genome and self.cash_init == other.cash_init and self.exchange == other.exchange:
            return True
        else:
            return False

    def got_enough_cash(self, value):
        if isinstance(value, (float, int)):
            if value >= 0:
                if value <= self.cash_current:
                    return True
                else:
                    print(f'Current cash ({self.cash_current}) not enough for value ({value})')
                    return False
            else:
                print(f'Negative value: {value}')
                return False
        else:
            raise TypeError('value not in (int, float)')

    def got_enough_amount(self, stock, amount):
        if isinstance(amount, (float, int)):
            if amount <= 0:
                if abs(amount) <= any(self.balance.loc[self.balance['stock_name'] == stock.name, 'amount']):
                    return True
                else:
                    print(f'Balance contains less amount of {stock.name}'
                          f' ({self.balance.loc[self.balance["stock_name"] == stock.name, "amount"]})'
                          f' than the required ({abs(amount)})')
                    return False
            else:
                print(f'Positive amount: {amount}')
                print('here1')
                return False
        elif isinstance(amount, pd.Series):
            if any(amount <= 0):
                if any(abs(amount)) <= any(self.balance.loc[self.balance['stock_name'] == stock.name, 'amount']):
                    return True
                else:
                    print(f'Balance contains less amount of {stock.name}'
                          f' ({self.balance.loc[self.balance["stock_name"] == stock.name, "amount"]})'
                          f' than the required ({abs(amount)})')
                    return False
            else:
                print(f'Positive amount: {amount}')
                print('here2')
                return False
        else:
            raise TypeError('value not in (int, float)')

    def get_stock_amount(self, stock):
        if self.balance.loc[self.balance['stock_name'] == stock.name, 'amount'].empty:
            return 0
        else:
            return self.balance.loc[self.balance['stock_name'] == stock.name, 'amount']


    def deduct_cash(self, amount):
        if isinstance(amount, (float, int)):
            if amount >= 0:
                self.cash_current -= amount

    def add_cash(self, amount):
        if isinstance(amount, (float, int)):
            if amount >= 0:
                self.cash_current += amount

    def update_cash_spent(self, value):
        """accepts negative value as well meaning that we get back cash by selling a stock"""
        if isinstance(value, (float, int)):
            self.cash_spent += value

    def update_total_portfolio_value(self, as_of):
        value_per_stock = []
        if self.balance.last_valid_index():
            for row in range(len(self.balance.index)):
                ticker = self.balance.iloc[row,self.balance.columns.get_loc('stock_name')]
                # check if ticker is np.nan which is a np.float object
                if not isinstance(ticker, np.float):
                    amount = self.balance.iloc[row,self.balance.columns.get_loc('amount')]
                    current_value = [stock.get_price(as_of) for stock in Stock.stock_list if stock.name == ticker]
                    value_per_stock.append(current_value[0] * amount)
                    self.total_portfolio_value = sum(value_per_stock)
        else:
            self.total_portfolio_value = self.cash_init

    def update_balance(self, stock, amount, price, value, sector=np.nan):
        non_zero = False
        if isinstance(np.round(abs(amount), 2), (int, float)):
            if not np.round(abs(amount), 2) == 0:
                non_zero = True
        elif isinstance(np.round(abs(amount), 2), pd.Series):
            if not any(np.round(abs(amount), 2) == 0):
                non_zero = True
        else:
            print('type(np.round(abs(amount),2))', type(np.round(abs(amount), 2)))
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
                if isinstance(last_idx, (int, float)):
                    self.balance.iloc[last_idx + 1, self.balance.columns.get_loc('stock_name')] = stock.name
                    self.balance.iloc[last_idx + 1, self.balance.columns.get_loc('amount')] = amount
                    self.balance.iloc[last_idx + 1,self.balance.columns.get_loc('price')] = price
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
        non_zero = False
        if isinstance(np.round(abs(amount), 2), (int, float)):
            if not np.round(abs(amount), 2) == 0:
                non_zero = True
        elif isinstance(np.round(abs(amount), 2), pd.Series):
            if not any(np.round(abs(amount), 2) == 0):
                non_zero = True
        else:
            print('type(np.round(abs(amount),2))', type(np.round(abs(amount), 2)))

        if non_zero:
            idx = self.log['stock_name'].last_valid_index()
            if isinstance(idx, (int, float)):
                try:
                    self.log.iloc[idx + 1, self.log.columns.get_loc('date')] = as_of
                    self.log.iloc[idx + 1, self.log.columns.get_loc('stock_name')] = stock.name
                    self.log.iloc[idx + 1, self.log.columns.get_loc('direction')] = direction
                    self.log.iloc[idx + 1, self.log.columns.get_loc('amount')] = amount
                    self.log.iloc[idx + 1, self.log.columns.get_loc('price')] = price
                    self.log.iloc[idx + 1, self.log.columns.get_loc('value')] = value
                except IndexError:
                    print('max index reached, iloc cannot expand the dataframe', self.log)

            elif idx is None:
                self.log.iloc[0, self.log.columns.get_loc('date')] = as_of
                self.log.iloc[0, self.log.columns.get_loc('stock_name')] = stock.name
                self.log.iloc[0, self.log.columns.get_loc('direction')] = direction
                self.log.iloc[0, self.log.columns.get_loc('amount')] = amount
                self.log.iloc[0, self.log.columns.get_loc('price')] = price
                self.log.iloc[0, self.log.columns.get_loc('value')] = value
            else:
                raise NotImplementedError

    def update_number_of_stocks(self):
        self.number_of_stocks = len(self.balance['stock_name'].unique())

    def buy(self, stock, amount, as_of=DATE):
        if not isinstance(stock, Stock):
            raise TypeError(f'Not Stock instance! type given: {type(stock)}')
        if not isinstance(as_of, datetime.date):
            raise TypeError(f'as_of should be datetime.date and not {type(as_of)}')

        if amount >= 0:
            price = stock.get_price(as_of)
            value = price * amount

            if self.got_enough_cash(value):
                self.deduct_cash(value)
                self.update_cash_spent(value)
                self.update_balance(stock, amount, price, value)
                self.update_log(as_of=as_of, stock=stock, direction='buy', amount=amount, price=price, value=value)

        else:
            raise ValueError('buy requires a positive amount')

    def sell(self, stock, amount, as_of=DATE):
        if not isinstance(stock, Stock):
            raise TypeError(f'Not Stock instance! type given: {type(stock)}')
        if isinstance(amount, pd.Series):
            if any(amount <= 0):
                price = stock.get_price(as_of)
                value = price * amount
        elif isinstance(amount, (int, float)):
            if amount <= 0:
                price = stock.get_price(as_of)
                value = price * amount
        else:
            raise ValueError('amount not int/float/pd.Series')

        if self.got_enough_amount(stock, amount):
            self.add_cash(abs(value))
            self.update_cash_spent(value)
            self.update_balance(stock, amount, price, value)
            self.update_log(as_of=as_of, stock=stock, direction='sell', amount=amount, price=price, value=value)
        else:
            print('Do not have enough amount to sell')
            pass
