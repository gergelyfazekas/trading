import time

import mysql.connector
import datetime
import pandas as pd
from pandas_datareader import data as wb
import stock_class
import tuning
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import exc


def sql_connect_old(host="database-1.c30doxhxuudc.us-east-1.rds.amazonaws.com", user="admin", database="trading_test"):
	db = mysql.connector.connect(
		host=host,
		user=user,
		passwd=str(input("MySQL database password:")),
		database=database
	)
	mycursor = db.cursor()
	return mycursor, db


def sql_disconnect_old(cursor, db):
	cursor.close()
	db.close()


def insert_data_into_sql_old(ticker_df, mycursor, db, sql_table="stock_prices"):
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


def price_query_sql_old(ticker_name, mycursor, db, start_date=datetime.date(1990, 1, 1), end_date=datetime.date.today(),
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


def get_unique_names_sql_old(mycursor, db, sql_table='stock_prices'):
	check_connection(mycursor, db)
	mycursor.execute(f"SELECT DISTINCT ticker FROM {sql_table}")
	queried_data = mycursor.fetchall()
	unique_names = tuning.flatten(queried_data, num_iter=1)
	return unique_names


def check_connection(mycursor, db):
	if not db.is_connected():
		mycursor, db = sql_connect()


def fill_sql_from_yahoo_old(mycursor, db, length=2, start_date=None, end_date=None, sql_table="stock_prices"):
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
			insert_data_into_sql(df=futures_item.result(), sql_table=sql_table)
		except KeyError:
			pass


def pull_all_data_sql_old(mycursor, db, sql_table='stock_prices', stock_names=None, set_each=True, return_whole=True):
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
	sql_names = get_unique_names_sql(mycursor, db, sql_table)
	if stock_names:
		sql_names = [ticker for ticker in stock_names if ticker in sql_names]
	stock_class.Stock.clear_stock_list()
	stock_class.Stock.create_stock_list_sql(sql_names)
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


# currently insert_data_into_sql is used for pushing data to sql
def push_data_sql_old(df, mycursor, db, sql_table='stock_prices'):
	"""updates existing sql_table with new columns from df

	arguments:
	df: a combined dataframe of multiple (all) stock's data possibly via Stock.aggregate_data
	sql_table: already existing sql table to be updated with new columns
	"""

	# get column names from existing sql_table
	mycursor.execute(f"SHOW COLUMNS FROM {sql_table}")
	database_columns = [x[0] for x in mycursor.fetchall()]

	# get column names that are not in the existing table but in the df
	new_cols = [col_name for col_name in df.columns if col_name not in database_columns]
	print(-2)
	# create a temporary table and fill it with the df['id']
	mycursor.execute(f"CREATE TABLE IF NOT EXISTS tmp (id int)")
	print(-1)
	for row in range(len(df.index)):
		print(row)
		mycursor.execute(f"INSERT INTO tmp (id) VALUES ({str(df.iloc[row, df.columns.get_loc('id')])})")
	print(0)
	# check if new_cols is not empty
	if new_cols:
		print(1)
		# for every column that we want to add get the type
		for idx, new_col_name in enumerate(new_cols):
			if isinstance(df.iloc[0, df.columns.get_loc(new_col_name)], int):
				new_col_type = 'INT'
			elif isinstance(df.iloc[0, df.columns.get_loc(new_col_name)], float):
				new_col_type = 'FLOAT'
			elif isinstance(df.iloc[0, df.columns.get_loc(new_col_name)], str):
				new_col_type = 'VARCHAR'
			elif isinstance(df.iloc[0, df.columns.get_loc(new_col_name)], datetime.date):
				new_col_type = 'DATE'
			elif isinstance(df.iloc[0, df.columns.get_loc(new_col_name)], bool):
				new_col_type = 'BOOLEAN'
			else:
				raise TypeError(
					f'type {type(df.iloc[0, df.columns.get_loc(new_col_name)])} not implemented in push_data_sql')

			print(2, new_col_name, new_col_type)
			# add the current column to the temporary table inside the loop
			mycursor.execute(f"ALTER TABLE tmp ADD {new_col_name} {new_col_type}")
			print(3)
			# fill the new sql column with values from df['new_col']
			for row in range(len(df.index)):
				print(row)
				mycursor.execute(
					f"INSERT INTO tmp ({new_col_name}) VALUES ({str(df.iloc[row, df.columns.get_loc(new_col_name)])})")
			print(4)
			# add the new_col from tmp to sql_table
			mycursor.execute(
				f"UPDATE new JOIN new2 USING(id) SET {sql_table}.{new_col_name}=tmp.{new_col_name}")
			print(5)
			# drop new_col from tmp so that during the loop it remains remains id, new_col
			mycursor.execute(f"ALTER TABLE tmp DROP COLUMN {new_col_name}")
			db.commit()
			print(6)


# for messing around
def sql_execute_old(conn, statement, values=None):
	"""general function executing any sql command for playing around in jupyter notebook"""
	if values:
		return conn.execute(statement, values).fetchall()
	else:
		return conn.execute(statement).fetchall()


def sql_connect(host="database-1.c30doxhxuudc.us-east-1.rds.amazonaws.com", user="admin", database="trading_test"):
	passwd = str(input("MySQL database password:"))
	engine = create_engine(
		f"mysql+mysqlconnector://{user}:{passwd}@{host}/{database}")

	conn = engine.connect()
	return conn


def insert_data_into_sql(df, engine, sql_table="stock_prices", if_exists="fail", create_backup=False, verbose=True):
	"""main function for pushing dataframe into sql

	either aggregate the data to a total_df and use if_exists='replace' or append stock by stock"""
	max_rows_at_once = 100000
	backup_table_name = "stock_prices_backup"
	if sql_table == "stock_prices":
		if if_exists == "replace":
			if not create_backup:
				print("stock_prices is the main table, do not use if_exists=replace with create_backup=False")
				raise InterruptedError
	if create_backup:
		if verbose:
			print(f"creating backup in table: {backup_table_name}")
		try:
			engine.execute(f'CREATE TABLE {backup_table_name} LIKE {sql_table}')
		except exc.ProgrammingError as err:
			error_msg = str(err.__dict__['orig'])
			if error_msg.endswith("exists") and error_msg.startswith("1050"):
				if verbose:
					print("Got error: error_msg")
					print("Dropping existing backup table and recreating from scratch")
				engine.execute(f'DROP TABLE {backup_table_name}')
				engine.execute(f'CREATE TABLE {backup_table_name} LIKE {sql_table}')
		engine.execute(f'INSERT INTO {backup_table_name} SELECT * FROM {sql_table}')
		if verbose:
			print("backup successful")
	if verbose:
		print(f"uploading df to {sql_table} with if_exists={if_exists}")
	# slice the df to a number of max_rows_at_once chunks and appending them to each other one-by-one
	i = 0
	while i < len(df.index):
		df_slice = df.iloc[i:i+max_rows_at_once,:].copy()
		df_slice.to_sql(name=sql_table, con=engine, if_exists=if_exists, index=False)
		if_exists = "append"
		i += max_rows_at_once


def create_sql_table(engine, sql_table, columns):
	if exists_sql_table(engine, sql_table):
		raise KeyError(f'sql_table {sql_table}'
					   f' already exists use fill_sql_from_yahoo/insert_data_into_sql with append')

	for idx, new_col_name in enumerate(columns):
		if isinstance(df.iloc[0, df.columns.get_loc(new_col_name)], int):
			new_col_type = 'INT'
		elif isinstance(df.iloc[0, df.columns.get_loc(new_col_name)], float):
			new_col_type = 'FLOAT'
		elif isinstance(df.iloc[0, df.columns.get_loc(new_col_name)], str):
			new_col_type = 'VARCHAR'
		elif isinstance(df.iloc[0, df.columns.get_loc(new_col_name)], datetime.date):
			new_col_type = 'DATE'
		elif isinstance(df.iloc[0, df.columns.get_loc(new_col_name)], bool):
			new_col_type = 'BOOLEAN'
		else:
			raise TypeError(
				f'type {type(df.iloc[0, df.columns.get_loc(new_col_name)])} not implemented in push_data_sql')

		if idx == 0:
			engine.execute(f'CREATE TABLE IF NOT EXISTS {sql_table} ({str(new_col_name)} {str(new_col_type)})')
		else:
			engine.execute(f'ALTER TABLE {sql_table} ADD {str(new_col_name)} {str(new_col_type)}')


def price_query_sql(ticker_name, conn, start_date=datetime.date(1990, 1, 1), end_date=datetime.date.today(),
					sql_table="stock_prices"):
	start_date = str(start_date)
	end_date = str(end_date)
	sql_table = str(sql_table)

	result = conn.execute(f"SHOW COLUMNS FROM {sql_table}")
	database_columns = [x[0] for x in result.fetchall()]

	quried_data = conn.execute(f'SELECT * FROM {sql_table} '
							   f'WHERE ticker = "{ticker_name}" '
							   f'AND date_ BETWEEN "{start_date}" '
							   f'AND "{end_date}"').fetchall()
	ticker_df = pd.DataFrame(data=quried_data, columns=database_columns)
	ticker_df.loc[:, 'date_'] = pd.to_datetime(ticker_df.loc[:, 'date_'])
	ticker_df.set_index('date_', inplace=True)
	# other functions might want to refer to 'date_' or index
	ticker_df['date_'] = ticker_df.index.copy()
	return ticker_df


def get_unique_names_sql(conn, sql_table='stock_prices', num_iter=0):
	queried_data = conn.execute(f"SELECT DISTINCT ticker FROM {sql_table}").fetchall()
	# wierd list of sqlalchemy objects has to be turned into a list of lists
	queried_data = [list(sublist) for sublist in queried_data]
	unique_names = tuning.flatten(queried_data, num_iter=num_iter)
	return unique_names


def exists_sql_table(conn, sql_table):
	try:
		conn.execute(f'SELECT * FROM {sql_table}').fetchall()
		return True
	except exc.SQLAlchemyError:
		return False


def fill_sql_from_yahoo(conn, length=2, start_date=None, end_date=None,
						sql_table="stock_prices", if_exists="fail", verbose=False, stock_csv="tickers_30.csv", sep=';'):
	if exists_sql_table(conn, sql_table):
		pass
	else:
		raise KeyError(f'sql_table {sql_table}'
					   f' not created yet, use create_sql_table and then fill_sql_from_yahoo with append')

	stock_class.Stock.clear_stock_list()
	stock_class.Stock.create_stock_list_from_csv(stock_csv, sep=sep)
	for stock in stock_class.Stock.stock_list:
		print(stock.name)
		if not start_date:
			last_date_sql = conn.execute(
				f"SELECT date_ FROM {sql_table} WHERE ticker='{stock.name}' ORDER BY date_ DESC LIMIT 1").fetchall()
			print('last_date_sql', last_date_sql)
			if not last_date_sql:
				# if ticker not is sql_table the above conn.execute returns []
				# use default start-end dates
				print('not found in sql -- default', stock.yahoo_pull_start_date)
				continue
			start = last_date_sql[0][0] + datetime.timedelta(days=1)
		if not end_date:
			end = start + datetime.timedelta(days=length)
		if verbose:
			print(f"{stock.name}: {start_date}:{end_date}")
		stock.set_yahoo_pull_start_date(start)
		stock.set_yahoo_pull_end_date(end)
		print('set_date 1', stock.yahoo_pull_start_date)

	futures = stock_class.Stock.yahoo_pull_data_for_stock_list()
	for futures_item in futures:
		df = futures_item.result()
		try:
			insert_data_into_sql(df=df, sql_table=sql_table, engine=conn, if_exists=if_exists)
		except KeyError:
			pass
	for stock in stock_class.Stock.stock_list:
		if stock.data.empty:
			continue
		stock.lowercase()
		stock.set_index()
		insert_data_into_sql(df=stock.data, sql_table=sql_table, engine=conn, if_exists=if_exists)


def pull_all_data_sql(conn, sql_table='stock_prices', stock_names=None, set_each=True, return_whole=True, verbose=False):
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
	sql_names = get_unique_names_sql(conn, sql_table, num_iter=1)
	if stock_names:
		sql_names = [ticker for ticker in stock_names if ticker in sql_names]
	stock_class.Stock.clear_stock_list()
	stock_class.Stock.create_stock_list_sql(sql_names)
	sql_names = tuple(sql_names)
	if verbose:
		print("pulling data")
	result = conn.execute(f"SHOW COLUMNS FROM {sql_table}").fetchall()
	database_columns = [x[0] for x in result]
	if stock_names:
		total_df = conn.execute(f"SELECT * FROM {sql_table} WHERE ticker IN {sql_names}").fetchall()
	else:
		# WHERE condition slows down -- takes twice the time -- avoid if possible
		total_df = conn.execute(f"SELECT * FROM {sql_table}").fetchall()
	total_df = pd.DataFrame(data=total_df, columns=database_columns)
	total_df.loc[:, 'date_'] = pd.to_datetime(total_df.loc[:, 'date_'])
	total_df.set_index('date_', inplace=True)
	total_df['date_'] = total_df.index.copy()
	if set_each:
		start_time = time.time()
		for stock in stock_class.Stock.stock_list:
			stock.set_data(total_df[total_df['ticker'] == stock.name].copy())
			tot_time = time.time()-start_time
			if verbose:
				print(stock.name, tot_time)
	if return_whole:
		return total_df


def main():
	mycursor, db = sql_connect()
	fill_sql_from_yahoo(mycursor, db)


if __name__ == "__main__":
	main()
