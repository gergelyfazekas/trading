import time
from tech_ind import sma_cross, sma_calc, rsi


def run_strategy(strategy_dict):
    for item in strategy_dict.items():
        item[0](*item[1])


def main():
    stocks = pull_data()
    stocks_keylist = list(stocks.keys())

    for ticker in stocks_keylist:
        strategy = {rsi: [stocks[ticker], 14],
                    sma_cross: [stocks[ticker], 14, 28, 5]}
        run_strategy(strategy)


if __name__ == '__main__':
    print("started")
    start = time.time()

    main()

    print("tottime: ", time.time() - start)
