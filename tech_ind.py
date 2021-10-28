import pandas as pd
import numpy as np


def sma_calc(stock, period=14):
    sma = stock.data.Close.rolling(window=period).mean()
    stock.data[f'sma_{period}'] = sma
    # return sma

def sma_cross(stock, sma_short_days, sma_long_days, max_days_since_cross):
    flag = -1
    if (sma_short_days and sma_long_days) in stock.data.keys():
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
            print('Not finished this part yet')

    else:
        sma_calc(stock=stock, period=sma_short_days)




def peak_finder():
    pass