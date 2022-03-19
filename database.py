import mysql.connector
import datetime
import pandas as pd
from pandas_datareader import data as wb
import stock_class


def sql_connect(host="localhost", user="root", database="trading_database"):
	db = mysql.connector.connect(
		host= host,
		user= user,
		passwd=str(input("MySQL database password:")),
		database= database
		)
	mycursor = db.cursor()
	return mycursor, db


def sql_disconnect(cursor, db):
	cursor.close()
	db.close()


def insert_data_into_sql(ticker_df, mycursor, db):

	if not isinstance(ticker_df, pd.DataFrame):
		raise TypeError('ticker_df: not pandas.dataframe')
		# return False

	check_connection(mycursor,db)

	data_to_list = ticker_df.values.tolist()
	for x in range(len(data_to_list)):
		tuple_values = tuple(data_to_list[x])
		mycursor.execute("INSERT INTO stock_prices (high, low, open, close, volume, adj_close, date_, ticker) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", tuple_values)
	db.commit()

	mycursor.execute("SELECT * FROM stock_prices")
	for x in mycursor:
		print(x)


def price_query_sql(ticker_name, start_date=datetime.date(1990,1,1), end_date=datetime.date.today(), sql_table="stock_prices"):

	check_connection(mycursor,db)

	mycursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA WHERE TABLE_NAME = {sql_table}")
	database_columns = [x[0] for x in mycursor.fetchall()]
	mycursor.execute(f"SELECT * FROM {sql_table} "
					 f"WHERE ticker = {ticker_name} "
					 f"AND date >= {start_date} "
					 f"AND date <= {end_date}")
	quried_data = mycursor.fetchall()
	ticker_df = pd.DataFrame(data=quried_data, columns=database_columns)
	return ticker_df


def check_connection(mycursor, db):
	if not db.is_connected():
		mycursor, db = sql_connect()


def fill_sql_from_yahoo(mycursor,db, length = 2, start_date = None, end_date = None):
	check_connection(mycursor,db)

	if start_date is None:
		mycursor.execute("SELECT date_ FROM stock_prices ORDER BY id DESC LIMIT 1")
		last_date = mycursor.fetchall()
		start_date = last_date[0][0].date() + datetime.timedelta(days = 1)

	if end_date is None:
		end_date = start_date + datetime.timedelta(days = length)

	stock_class.Stock.yahoo_pull_start_date = start_date
	stock_class.Stock.yahoo_pull_end_date = end_date

	stock_class.Stock.create_stock_list_from_csv()

	futures = stock_class.Stock.yahoo_pull_data_for_stock_list()
	for futures_item in futures:
		insert_data_into_sql(futures_item.result(), mycursor, db)


def main():
	mycursor, db = sql_connect()

	fill_sql_from_yahoo(mycursor,db)


	# if 'database' in sys.modules:
		#print('Import successful')
	# else:
		#print('Import not successful')

if __name__ == "__main__":
    main()


