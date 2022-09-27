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
    portfolio_list = []

    def __init__(self, genome, cash):
        if isinstance(cash, (int, float)):
            __class__.portfolio_list.append(self)
            self.exchange = "NASDAQ"
            self.genome = genome
            self.cash = cash
            self.current_invested_value = 0
            self.number_of_stocks = int
            self.sectors = list
            self.currencies = list
            self.log = pd.DataFrame({'date': [np.nan]*stock_class.PLACEHOLDER,
                                     'stock_name': [np.nan]*stock_class.PLACEHOLDER,
                                     'direction': [np.nan]*stock_class.PLACEHOLDER,
                                     'amount': [np.nan]*stock_class.PLACEHOLDER,
                                     'price': [np.nan]*stock_class.PLACEHOLDER,
                                     'value': [np.nan]*stock_class.PLACEHOLDER})
            self.balance = pd.DataFrame({'stock_name': [np.nan]*stock_class.PLACEHOLDER,
                                         'amount': [np.nan]*stock_class.PLACEHOLDER,
                                         'price': [np.nan]*stock_class.PLACEHOLDER,
                                         'value': [np.nan]*stock_class.PLACEHOLDER,
                                         'sector': [np.nan]*stock_class.PLACEHOLDER})
            self.total_portfolio_value = self.cash + self.balance['value'].sum()
            self.proportion_invested = self.current_invested_value / self.total_portfolio_value

    def __eq__(self, other):
        if self.genome == other.genome and self.cash == other.cash and self.exchange == other.exchange:
            return True
        else:
            return False

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

    def got_enough_amount(self, stock, amount):
        if isinstance(amount, (float, int)):
            if amount <= 0:
                if abs(amount) <= self.balance[self.balance['stock_name'] == stock.name]['amount'][0]:
                    return True
                else:
                    print(f'Balance contains less amount of {stock.name}'
                          f' ({self.balance[balance["stock_name"] == stock.name]["amount"]})'
                          f' than the required ({abs(amount)})')
                    return False
            else:
                print(f'Positive amount: {amount}')
                return False
        else:
            raise TypeError('value not in (int, float)')

    def get_stock_amount(self, stock):
        return self.balance.loc[self.balance['stock_name'] == stock.name, 'amount']


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

    def update_balance(self, stock, amount, price, value, sector=np.nan):
        # if stock is in the list already
        if stock.name in list(self.balance['stock_name']):
            idx = list(self.balance['stock_name']).index(stock.name)

            # update
            self.balance['amount'].iloc[idx] = self.balance['amount'].iloc[idx] + amount
            self.balance['price'].iloc[idx] = price
            self.balance['value'].iloc[idx] = self.balance['price'].iloc[idx] * self.balance['amount'].iloc[idx]


        else:
            # if stock is not in the list yet put it in the last row
            idx = self.balance['stock_name'].last_valid_index()
            if isinstance(idx, (int, float)):
                self.balance.loc[idx + 1, 'stock_name'] = stock.name
                self.balance.loc[idx + 1, 'amount'] = amount
                self.balance.loc[idx + 1, 'price'] = price
                self.balance.loc[idx + 1, 'value'] = value
                self.balance.loc[idx + 1, 'sector'] = sector

            else:
                # if idx is None then put it in the first row
                self.balance.loc[0, 'stock_name'] = stock.name
                self.balance.loc[0, 'amount'] = amount
                self.balance.loc[0, 'price'] = price
                self.balance.loc[0, 'value'] = value
                self.balance.loc[0, 'sector'] = sector

    def update_log(self,date, stock, direction, amount, price, value):
        idx = self.log['stock_name'].last_valid_index()
        if isinstance(idx, (int, float)):
            self.log.loc[idx + 1, 'date'] = date
            self.log.loc[idx + 1, 'stock_name'] = stock.name
            self.log.loc[idx + 1, 'direction'] = direction
            self.log.loc[idx + 1, 'amount'] = amount
            self.log.loc[idx + 1, 'price'] = price
            self.log.loc[idx + 1, 'value'] = value
        else:
            self.log.loc[0, 'date'] = date
            self.log.loc[0, 'stock_name'] = stock.name
            self.log.loc[0, 'direction'] = direction
            self.log.loc[0, 'amount'] = amount
            self.log.loc[0, 'price'] = price
            self.log.loc[0, 'value'] = value

    def update_number_of_stocks(self):
        self.number_of_stocks = len(self.balance['stock_name'].unique())


    def buy(self, stock, amount, date = DATE):
        if not isinstance(stock, Stock):
            raise TypeError(f'Not Stock instance! type given: {type(stock)}')
        if not isinstance(date, datetime.date):
            raise TypeError(f'date should be datetime.date and not {type(date)}')

        if amount >= 0:
            price = stock.get_price(date)
            value = price * amount

            if self.got_enough_cash(value):
                self.deduct_cash(value)
                self.update_current_invested_value(value)
                self.update_balance(stock, amount, price, value)
                self.update_log(date=date, stock=stock, direction='buy',
                                amount=amount, price=price, value=value)

        else:
            raise ValueError('buy requires a positive amount')

    def sell(self, stock, amount, date=DATE):
        if not isinstance(stock, Stock):
            raise TypeError(f'Not Stock instance! type given: {type(stock)}')

        if amount <= 0:
            price = stock.get_price(date)
            value = price * amount

            if self.got_enough_amount(stock, amount):
                self.add_cash(abs(value))
                self.update_current_invested_value(value)
                self.update_balance(stock, amount, price, value)
                self.update_log(date=date, stock=stock, direction='sell',
                                amount=amount, price=price, value=value)

        else:
            raise ValueError('sell requires a negative amount')









