import pandas as pd
import numpy as np


def sma_calc(stock, period=14):
    sma = stock.data.Close.rolling(window=period).mean()
    stock.data[f'sma_{period}'] = sma
    # return sma

def sma_cross(stock, sma_short_days, sma_long_days, max_days_since_cross):
    if f'sma_{sma_long_days}' in stock.data.columns:
        if f'sma_{sma_short_days}' in stock.data.columns:
            sma_short = stock.data[f'sma_{sma_short_days}']
            sma_long = stock.data[f'sma_{sma_long_days}']

            below_or_above = np.where(stock.data[sma_short] < stock.data.[sma_long], 0,
                                     np.where(stock.data[sma_short] > stock.data.[sma_long], 1, np.nan))

            crosses = [0]
            for i in range(len(below_or_above)-1):
                crosses.append(below_or_above[i + 1] - below_or_above[i])

            if np.argmax(crosses.reverse()) <= max_days_since_cross:
                stock.set_sma_cross()

        else:
            sma_calc(stock=stock, period=sma_short_days)
            sma_cross(stock=stock, sma_short_days=sma_short_days, sma_long_days=sma_long_days,
                      max_days_since_cross=max_days_since_cross)
    else:
        sma_calc(stock=stock, period=sma_long_days)
        sma_cross(stock=stock, sma_short_days=sma_short_days, sma_long_days=sma_long_days,
                  max_days_since_cross=max_days_since_cross)


def rsi(stock, lookback):
    values = stock.data.Close[-lookback:]
    up = values[values>0].mean()
    down = -1*values[values<0].mean()
    return 100 * up / (up + down)


def macd(stock, lookback):
    pass


def peak_finder():
    pass