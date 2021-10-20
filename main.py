#improved version of Python_Trading_v5
import datetime

#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from pandas_datareader import data as wb

class Stock:
    def __init__(self,name):
        self.name = name
        self.analysis_period = 100
        #self.data = pd.DataFrame


    def get_data(self, date):
        start_date = date - datetime.timedelta(self.analysis_period)
        self.data = wb.DataReader(self.name, "yahoo", start_date, date)





def get_tickers():
    tickers = open("tickers.txt","r").read()
    return tickers

