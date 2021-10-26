import pandas as pd
import numpy as np


def sma_calc(stock, period=14):
    sma = stock.Close.rolling(window=period).mean()
    return sma

def sma_cross(stock, sma_short_days, sma_long_days):
    flag = -1
    if (sma_short_days and sma_long_days) in stock.data.keys():
        sma_short = f'sma_{sma_short_days}'
        sma_long = f'sma_{sma_long_days}'

        crossing_list = np.where(stock.data[sma_short] < stock.data.[sma_long], 0,
                                 np.where(stock.data[sma_short] > stock.data.[sma_long], 1, np.nan))

        crossing_points = [1 for i in range(len(crossing_list)) if crossing_list[i] < crossing_list[i+1]]

    else:
        sma_calc(stock=stock, period=sma_short)




def peak_finder():
    pass