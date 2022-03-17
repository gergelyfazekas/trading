import concurrent.futures
import datetime
from pandas_datareader import data as wb
import pandas as pd

# Constants
MAX_WORKERS = 20
NUMBER_OF_TICKERS = 20
ANALYSIS_PERIOD = 20


class Stock:
    stock_list = []
    yahoo_pull_start_date = datetime.date()
    yahoo_pull_end_date = datetime.date()

    @staticmethod
    def create_top_tickers_csv(filename="nasdaq_tickers.csv", number=30, sort_by="Market Cap", asc=False):
        tickers = pd.read_csv(open(filename, encoding="latin-1"))
        tickers = tickers.sort_values(by=sort_by, ascending=asc)
        tickers_by_ipo = tickers.loc[(pd.notnull(tickers['IPO Year'])) & (tickers['IPO Year'] <= 2006)]
        tickers_30 = tickers_by_ipo.iloc[0:number, :]
        tickers_30.to_csv(f"tickers_{number}.csv")
        print(f"tickers_{number}.csv was created")

    @classmethod
    def create_stock_list_from_csv(cls, filename="nasdaq_tickers.csv"):
        excel_data = pd.read_csv(open(filename, encoding="latin-1"))
        tickers = excel_data['Symbol'].copy()
        for ticker in tickers[:NUMBER_OF_TICKERS]:
            cls.stock_list.append(Stock(name=ticker))

    @classmethod
    def print_stocks(cls):
        ticker_lst = []
        for stock in cls.stock_list:
            ticker_lst.append(stock.name)
        print(ticker_lst)

    @classmethod
    def get_stock_names(cls):
        ticker_lst = []
        for stock in cls.stock_list:
            ticker_lst.append(stock.name)
        return ticker_lst

    @classmethod
    def yahoo_pull_data_for_stock_list(cls):
        if len(cls.stock_list) > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(ticker.set_data(ticker.yahoo_pull_data)) for ticker in cls.stock_list]
                concurrent.futures.wait(futures)

            no_data_tickers = [item for item in cls.stock_list if
                               item.data.empty]

            cls.pop_no_data_tickers()
            print("No data found (and removed from dict): ", no_data_tickers)

    @classmethod
    def pop_no_data_tickers(cls):
        cls.stock_list = [item for item in cls.stock_list if not item.data.empty]

    def __init__(self, name):
        self.name = name
        self.__class__.stock_list.append(self)
        self.analysis_period = ANALYSIS_PERIOD
        self.data = pd.DataFrame
        self.sma_cross = False

    def yahoo_pull_data(self):
        yahoo_data = wb.DataReader(self.name, "yahoo", self.start_date, self.end_date)
        yahoo_data["Date"] = yahoo_data.index.strftime('%Y-%m-%d %X')
        return yahoo_data

    def set_data(self, new_dataframe):
        self.data = new_dataframe

    def set_yahoo_pull_start_date(self):
        # There might not be a need for specific date settings for different tickers
        pass

    def set_yahoo_pull_end_date(self):
        # There might not be a need for specific date settings for different tickers
        pass

    def set_sma_cross(self):
        self.sma_cross = True
