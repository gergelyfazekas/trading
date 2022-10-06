import mysql.connector
import datetime
import pandas as pd
from pandas_datareader import data as wb
import stock_class
import tuning


def sql_connect(host="database-1.c30doxhxuudc.us-east-1.rds.amazonaws.com", user="admin", database="trading_test"):
	db = mysql.connector.connect(
		host=host,
		user=user,
		passwd=str(input("MySQL database password:")),
		database=database
	)
	mycursor = db.cursor()
	return mycursor, db


def sql_disconnect(cursor, db):
	cursor.close()
	db.close()


def insert_data_into_sql(ticker_df, mycursor, db, sql_table="stock_prices"):
	if not isinstance(ticker_df, pd.DataFrame):
		raise TypeError('ticker_df: not pandas.dataframe')
	# return False

	check_connection(mycursor, db)

	data_to_list = ticker_df.values.tolist()
	for x in range(len(data_to_list)):
		tuple_values = tuple(data_to_list[x])
		mycursor.execute(
			f"INSERT INTO {sql_table} (high, low, open, close, volume, adj_close, date_, ticker) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
			tuple_values)
	db.commit()

	mycursor.execute(f"SELECT * FROM {sql_table}")
	for x in mycursor:
		print(x)


def price_query_sql(ticker_name, mycursor, db, start_date=datetime.date(1990, 1, 1), end_date=datetime.date.today(),
					sql_table="stock_prices"):
	start_date = str(start_date)
	end_date = str(end_date)
	sql_table = str(sql_table)

	check_connection(mycursor, db)
	mycursor.execute(f"SHOW COLUMNS FROM {sql_table}")
	database_columns = [x[0] for x in mycursor.fetchall()]

	mycursor.execute(f'SELECT * FROM {sql_table} '
					 f'WHERE ticker = "{ticker_name}" '
					 f'AND date_ BETWEEN "{start_date}" '
					 f'AND "{end_date}"')
	quried_data = mycursor.fetchall()
	ticker_df = pd.DataFrame(data=quried_data, columns=database_columns)
	ticker_df.loc[:, 'date_'] = pd.to_datetime(ticker_df.loc[:, 'date_'])
	ticker_df.set_index('date_', inplace=True)
	# other functions might want to refer to 'date_' or index
	ticker_df['date_'] = ticker_df.index
	return ticker_df


def get_unique_names_sql(mycursor, db, sql_table='stock_prices'):
	check_connection(mycursor, db)
	mycursor.execute(f"SELECT DISTINCT ticker FROM {sql_table}")
	queried_data = mycursor.fetchall()
	unique_names = tuning.flatten(queried_data, num_iter=1)
	return unique_names


def check_connection(mycursor, db):
	if not db.is_connected():
		mycursor, db = sql_connect()


def fill_sql_from_yahoo(mycursor, db, length=2, start_date=None, end_date=None, sql_table="stock_prices"):
	check_connection(mycursor, db)

	if start_date is None:
		mycursor.execute(f"SELECT date_ FROM {sql_table} ORDER BY id DESC LIMIT 1")
		last_date = mycursor.fetchall()
		start_date = last_date[0][0] + datetime.timedelta(days=1)

	if end_date is None:
		end_date = start_date + datetime.timedelta(days=length)

	stock_class.Stock.yahoo_pull_start_date = start_date
	stock_class.Stock.yahoo_pull_end_date = end_date

	stock_class.Stock.create_stock_list_from_csv()

	futures = stock_class.Stock.yahoo_pull_data_for_stock_list()
	for futures_item in futures:
		try:
			insert_data_into_sql(futures_item.result(), mycursor, db)
		except KeyError:
			pass


def pull_all_data_sql(mycursor, db, sql_table='stock_prices', stock_names=None, set_each=True, return_whole=True):
	"""pulls all data form a given sql table EVEN IF stock_names IS NOT None!!!
	(1) sets it to stock.data (if set_data=True)
	(2) returns the whole sql table as a df (if return_whole = True)
	if setting is enough use stock_names=None, return_whole=False
	if setting for selected stocks is enough use stock_names=['AAPL', 'MSFT'], return_whole=False

	parameters:
	myc, db: standard database.py related args, see sql_connect and price_query_sql functions
	sql_table: string in single quotes
	stock_names: a list, if specified only the elements of this list are selected, if None all stocks in sql_table are
	set_each: for each stock sets stock.data, default:True
	return_whole: returns a df containing all stock data from sql_table regardless of stock_names, default:True
	"""
	if not isinstance(sql_table, str):
		sql_table = str(sql_table)
	sql_names = get_unique_names_sql(mycursor, db)
	if stock_names:
		sql_names = [ticker for ticker in stock_names if ticker in sql_names]
		print(sql_names)
	stock_class.Stock.clear_stock_list()
	print(stock_class.Stock.stock_list)
	stock_class.Stock.create_stock_list_sql(sql_names)
	print('stock_list again', stock_class.Stock.stock_list)
	if set_each:
		for stock in stock_class.Stock.stock_list:
			stock.set_data(price_query_sql(stock.name, mycursor, db))
	if return_whole:
		mycursor.execute(f"SHOW COLUMNS FROM {sql_table}")
		database_columns = [x[0] for x in mycursor.fetchall()]
		mycursor.execute(f"SELECT * FROM {sql_table}")
		total_df = mycursor.fetchall()
		total_df = pd.DataFrame(data=total_df, columns=database_columns)
		total_df.loc[:, 'date_'] = pd.to_datetime(total_df.loc[:, 'date_'])
		total_df.set_index('date_', inplace=True)
		total_df['date_'] = total_df.index
		return total_df


def push_data_sql(df, mycursor, db, sql_table='stock_prices'):
	mycursor.execute(f"SHOW COLUMNS FROM {sql_table}")
	database_columns = [x[0] for x in mycursor.fetchall()]

	new_cols = [col_name for col_name in df.columns if col_name not in database_columns]
	if new_cols:
		col_types = []
		for idx, new_col in enumerate(new_cols):
			if isinstance(df.iloc[0, df.columns.get_loc(new_col)], int):
				col_types.append('INT')
			elif isinstance(df.iloc[0, df.columns.get_loc(new_col)], float):
				col_types.append('FLOAT')
			elif isinstance(df.iloc[0, df.columns.get_loc(new_col)], str):
				col_types.append('VARCHAR')
			elif isinstance(df.iloc[0, df.columns.get_loc(new_col)], datetime.date):
				col_types.append('DATE')
			elif isinstance(df.iloc[0, df.columns.get_loc(new_col)], bool):
				col_types.append('BOOLEAN')
			else:
				raise TypeError(f'type {type(df.iloc[0, df.columns.get_loc(new_col)])} not implemented in push_data_sql')
			mycursor.execute(f"ALTER TABLE {sql_table} ADD {new_col} {col_types[idx]}")
			# maybe create a new table with column names already including new_col
			# rename the old table to somthing and name the new table 'stock_prices'
			# insert the data (like insert_data_into_sql) the only difference is that col names should also be %s, %s ..
			pass

	for new_col in new_cols:
		mycursor.execute(f"SHOW")

	pass


def main():
	mycursor, db = sql_connect()
	fill_sql_from_yahoo(mycursor, db)


if __name__ == "__main__":
	main()
