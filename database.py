import mysql.connector
import pandas as pd
import numpy as np
from pandas_datareader import data as wb
import concurrent.futures
import yfinance as yf



def sql_connect(host="localhost", user="root", database="trading_database"):
	db = mysql.connector.connect(
		host= host,
		user= user,
		passwd=str(input("MySQL databse password:")),
		database= database
		)

	mycursor = db.cursor()
	return mycursor, db


#Choose 30 stocks for training 

def read_tickers_from_csv():
    tickers = pd.read_csv(open("nasdaq_tickers.csv",encoding="latin-1"))
    return tickers

def get_top_30_tickers():
	tickers = read_tickers_from_csv()
	tickers = tickers.sort_values(by = "Market Cap", ascending = False)
	tickers_filtered_by_IPO = tickers.loc[(pd.notnull(tickers['IPO Year'])) & (tickers['IPO Year'] <= 2006)]
	tickers_30 = tickers_filtered_by_IPO.iloc[0:30,:]
	tickers_30.to_csv("tickers_30.csv")


def sql_disconnect(cursor, db):
	cursor.close()
	db.close()







# fb = yf.Ticker("FB")

# fb_info = fb.info

# print(fb_info)


#Creating stock_prices table




