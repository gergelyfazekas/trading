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

#Get stock prices data for tickers and insert it into the sql database 
#format of start and end: "2006-01-01"
def get_data_list(ticker, start_date, end_date):
	stock_price = wb.DataReader(f"{ticker}", "yahoo", start = start_date, end = end_date)
	stock_price["Date"] = stock_price.index.strftime('%Y-%m-%d %X')
	stock_price_list = stock_price.values.tolist()
	return stock_price_list



def insert_data_into_sql(ticker, list):
	mycursor, db = sql_connect()

	for x in range(len(list)): 
		tuple_values = tuple(list[x])
		mycursor.execute("INSERT INTO stock_prices (ticker, high, low, open, close, volume, adj_close, date_) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", (f"{ticker}",) + tuple_values)

	db.commit()

	mycursor.execute("SELECT * FROM stock_prices")

	for x in mycursor:
		print(x)

	sql_disconnect(mycursor,db)






# ticker_data = get_data_list("AAPL", "2006-01-06", "2006-01-10")

# insert_data_into_sql("AAPL", ticker_data)







