import concurrent.futures
import datetime
from pandas_datareader import data as wb
import pandas as pd
import numpy as np
import database
import sys
import matplotlib.pyplot as plt

import stock_class
from stock_class import Stock

DATE = datetime.date(2007, 1, 12)

class Portfolio:
    created = False

    def __init__(self, cash):
        if isinstance(cash, (int, float)):
            if not self.__class__.created:
                self.__class__.created = True
                self.cash = cash
                self.current_invested_value = 0
                self.total_portfolio_value = self.cash + self.current_invested_value
                self.log = pd.DataFrame({'date': [np.nan]*stock_class.PLACEHOLDER,
                                         'stock_name': [np.nan]*stock_class.PLACEHOLDER,
                                         'direction': [np.nan]*stock_class.PLACEHOLDER,
                                         'amount': [np.nan]*stock_class.PLACEHOLDER,
                                         'price': [np.nan]*stock_class.PLACEHOLDER,
                                         'value': [np.nan]*stock_class.PLACEHOLDER})
                self.balance = pd.DataFrame({'stock_name': [np.nan]*stock_class.PLACEHOLDER,
                                             'amount': [np.nan]*stock_class.PLACEHOLDER,
                                             'price': [np.nan]*stock_class.PLACEHOLDER,
                                             'value': [np.nan]*stock_class.PLACEHOLDER})



    def got_enough_cash(self, value):
        if isinstance(value, (float, int)):
            if value >= 0:
                if value <= self.cash:
                    return True
                else:
                    print(f'Cash ({self.cash}) not enough for value ({value})')
                    return False
            else:
                print(f'Negative value: {value}')
                return False
        else:
            raise TypeError('value not in (int, float)')

    def deduct_cash(self, amount):
        if isinstance(amount, (float, int)):
            if amount >= 0:
                self.cash = self.cash - amount

    def add_cash(self, amount):
        if isinstance(amount, (float, int)):
            if amount >= 0:
                self.cash = self.cash + amount

    def update_current_invested_value(self, value):
        self.current_invested_value = self.current_invested_value + value

    def update_balance(self, stock, amount, price, value):
        if stock.name in list(self.balance['stock_name']):
            idx = list(self.balance['stock_name']).index(stock.name)

            # update
            self.balance['amount'].iloc[idx] = self.balance['amount'].iloc[idx] + amount
            self.balance['price'].iloc[idx] = price
            self.balance['price'].iloc[idx] = self.balance['price'].iloc[idx] * self.balance['amount'].iloc[idx]

        else:
            idx = self.balance['stock_name'].last_valid_index()
            if isinstance(idx, (int, float)):
                self.balance.loc[idx + 1, 'stock_name'] = stock.name
                self.balance.loc[idx + 1, 'amount'] = amount
                self.balance.loc[idx + 1, 'price'] = price
                self.balance.loc[idx + 1, 'value'] = value

            else:
                # if idx is None then put it in the first row
                self.balance.loc[0, 'stock_name'] = stock.name
                self.balance.loc[0, 'amount'] = amount
                self.balance.loc[0, 'price'] = price
                self.balance.loc[0, 'value'] = value


    def buy(self, stock, amount, date = DATE):
        if not isinstance(stock, Stock):
            raise TypeError(f'Not Stock instance! type given: {type(stock)}')

        if amount >= 0:
            price = stock.get_price(date)
            value = price * amount

            if self.got_enough_cash(value):
                self.deduct_cash(value)
                self.update_current_invested_value(value)
                self.update_balance(stock, amount, price, value)

        else:
            raise ValueError('Negative amount. Use Portfolio.sell() instead.')









