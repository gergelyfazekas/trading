#!/usr/bin/env python
# coding: utf-8

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.ticktype import TickTypeEnum

import threading
import time
import numpy as np
import pandas as pd




class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.error = self.error
        self.data = None
        self.req_id = 0
        self.done = False
        self.nextOrderId = None
        self.order = None
        self.contract = None

    @staticmethod
    def error(self, reqId, errorCode, errorString=None):
        if errorCode == 202:
            print('order canceled')

    def tickPrice(self, reqId, tickType, price, attrib):
        # print('ticker id: ', reqId, 'tickType: ', TickTypeEnum.to_str(tickType), 'price: ', price)
        if tickType == 67 and reqId == self.req_id:
            self.data = price

    def nextValidId(self, orderId):
        self.nextOrderId = orderId
        self.trade()

    def trade(self):
        if self.order and self.contract:
            print("placing order")
            self.placeOrder(self.nextOrderId, self.contract, self.order)
            print("placed order")
        else:
            pass

    def orderStatus(self, orderId, status, filled,
                    remaining, avgFillPrice, permId,
                    parentId, lastFillPrice, clientId,
                    whyHeld, mktCapPrice):
        print("OrderStatus. Id:", orderId, "Status:", status, "Filled:", filled,
              "Remaining:", remaining, "AvgFillPrice:", avgFillPrice)

    def openOrder(self, orderId, contract, order, orderState):
        print("OpenOrder.", orderId, contract, order, orderState)

    def execDetails(self, reqId, contract, execution):
        print("Exec details. ", reqId, contract.symbol, contract.secType, contract.currency,
              execution.execId, execution.orderId, execution.shares)

    # def tickSize(self, tickerId, field, size):
    #     print("tickSize", size, tickerId, field)

    # if tickType == 68:
    # print('The current last price is: ', price)

    def waiting_for_result(self):
        if self.data is not None:
            return self.data
        else:
            time.sleep(0.1)
            return self.waiting_for_result()

    def run_loop(self):
        self.run()

    def stop(self):
        self.done = True
        self.disconnect()


def create_order(direction, quantity, order_type='MKT', limit_price=0):
    """creates the order object:
    args:
    direction: 'BUY' or 'SELL',
    quantity: int or float, quantity to buy or sell
    order_type: 'LMT' for limit or 'MKT' for market order
    limit_price: optional float, only considered if order_type == 'LMT'
    """
    # Create order object
    order = Order()
    order.eTradeOnly = False
    order.firmQuoteOnly = False
    order.action = direction
    order.totalQuantity = quantity
    order.orderType = order_type
    order.lmtPrice = limit_price
    return order

def execute_order(contract, order):
    app = IBapi()
    app.nextOrderId = 0
    app.contract = contract
    app.order = order
    print("Reconnecting")
    app.connect('127.0.0.1', 4002, 123)

    # # Start the socket in a thread
    # api_thread = threading.Thread(target=app.run_loop, daemon=True)
    # api_thread.start()
    threading.Timer(3, app.stop).start()
    app.run_loop()


def create_contract(ticker):
    if isinstance(ticker, str):
        # Create contract object from ticker name
        contract = Contract()
        contract.symbol = ticker
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        contract.primaryExchange = "NASDAQ"
        return contract
    else:
        raise ValueError(f"ticker must be str and not {type(ticker)}")


def create_contract_list(ticker_list):
    contract_list = []
    for ticker in ticker_list:
        contract_list.append(create_contract(ticker))
    return contract_list


def get_price_interactive_brokers(contract_list):
    app = IBapi()
    app.connect('127.0.0.1', 4002, 123)

    ticker_list = [item.symbol for item in contract_list]
    result_dict = dict(zip(ticker_list, [np.nan] * len(ticker_list)))

    # Start the socket in a thread
    api_thread = threading.Thread(target=app.run_loop, daemon=True)
    api_thread.start()

    time.sleep(1)  # Sleep interval to allow time for connection to server

    req_id = 0
    for contract in contract_list:
        # Request Market Data
        app.req_id = req_id
        app.reqMarketDataType(4)
        app.reqMktData(req_id, contract, '', False, False, [])
        result = app.waiting_for_result()
        app.data = None
        result_dict[contract.symbol] = result
        req_id += 1

    app.disconnect()
    return result_dict


if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'MSCI']
    contract_list = create_contract_list(tickers)

    # Query the current prices
    price = get_price_interactive_brokers(contract_list)
    print("Successful query.")
    print(f"Price: ", price)

    # Execute an order
    # contract = create_contract('AAPL')
    # order = create_order('SELL', 1, 'LMT', 141)
    # execute_order(contract, order)






