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


def insert_data_into_sql(ticker_df, db, mycursor):

	if not isinstance(ticker_df, pd.DataFrame):
		raise TypeError('ticker_df: not pandas.dataframe')
		# return False

	if not db.is_connected():
		mycursor, db = sql_connect()

	data_to_list = ticker_df.values.tolist()
	for x in range(len(data_to_list)):
		tuple_values = tuple(data_to_list[x])
		mycursor.execute("INSERT INTO stock_prices (high, low, open, close, volume, adj_close, date_, ticker) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", tuple_values)
	db.commit()

	mycursor.execute("SELECT * FROM stock_prices")
	for x in mycursor:
		print(x)


def price_query_sql(ticker_name, start_date=datetime.date(1990,1,1), end_date=datetime.date.today(), sql_table="stock_prices"):
	mycursor, db = sql_connect()

	mycursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA WHERE TABLE_NAME = {sql_table}")
	database_columns = [x[0] for x in mycursor.fetchall()]
	mycursor.execute(f"SELECT * FROM {sql_table} "
					 f"WHERE ticker = {ticker_name} "
					 f"AND date >= {start_date} "
					 f"AND date <= {end_date}")
	quried_data = mycursor.fetchall()
	ticker_df = pd.DataFrame(data=quried_data, columns=database_columns)
	return ticker_df



	def main():

    mycursor, db = sql_connect()


    stock_class.Stock.create_stock_list_from_csv()
    futures = stock_class.Stock.yahoo_pull_data_for_stock_list()




    for futures_item in futures:
        insert_data_into_sql(futures_item.result(), db, mycursor)



    # if 'database' in sys.modules:
    #     print('Import successful')
    # else:
    #     print('Import not successful')




if __name__ == "__main__":
    main()






# ticker_data = get_data_list("AAPL", "2006-01-06", "2006-01-10")
# insert_data_into_sql("AAPL", ticker_data)
