import neat
import pandas as pd
import numpy as np
import brain
import pickle
from stock_class import Stock
from portfolio_class import Portfolio
import random


def main():
    with open("validation.pickle", "rb") as f:
        validation = pickle.load(f)

    with open("best_genome.pickle", "rb") as f:
        best_genome = pickle.load(f)

    # Load configuration.
    config_file = r'neat_config.txt'

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # X_cols and portfolio_attributes
    X_cols = ['cat_1', 'cat_2', 'cat_3', 'Basic Materials', 'Communication Services',
              'Consumer Cyclical', 'Consumer Defensive', 'Energy',
              'Financial Services', 'Healthcare', 'Industrials', 'Real Estate',
              'Technology', 'Utilities', 'variance_100', 'variance_global']
    portfolio_attributes = ["proportion_invested", "entropy_stock", "entropy_sector"]

    # preprocessing
    training_data = validation.copy()
    brain.shape_config_file(config_file, X_cols, portfolio_attributes)
    X_cols = [col.lower() for col in X_cols]
    cash = 1000
    threshold = 0.05

    # create 1-member population
    p = neat.Population(config)
    p.population = {1: best_genome}

    # clear and set Stocks
    Stock.clear_stock_list()
    Stock.set_each(training_data)

    # same loop as in brain.run_neat
    for genome_id, genome in p.population.items():
        genome.fitness = 0
        genome_portfolio = Portfolio(genome=genome, cash=cash)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for current_date in training_data.index.unique():

            # shuffle
            random.shuffle(Stock.stock_list)
            for stock in Stock.stock_list:
                try:
                    df = stock.data.loc[current_date, X_cols]
                    for item in portfolio_attributes:
                        df[item] = genome_portfolio.__getattribute__(item)
                except KeyError:
                    print("KeyError -- continue")
                    continue
                output = net.activate(list(df))
                decision, proportion = brain.output_to_decision(output[0], threshold=threshold)
                if decision == 'buy':
                    amount = (proportion * genome_portfolio.cash_current) / stock.data.loc[current_date, "close"]
                    genome_portfolio.buy(stock=stock, amount=amount, as_of=current_date)
                elif decision == 'sell':
                    amount = -1 * (proportion * genome_portfolio.get_stock_amount(stock))
                    genome_portfolio.sell(stock=stock, amount=amount, as_of=current_date)
                elif not decision:
                    pass
                else:
                    raise NotImplementedError
            # End-of-day update
            genome_portfolio.update_balance_eod(as_of=current_date)
            genome.fitness += (genome_portfolio.portfolio_return_hist[-1][1] - 1) * 200
        genome.fitness += 4 * (min(genome_portfolio.entropy_stock, 2.3) + genome_portfolio.entropy_sector)

    # connection weights
    X_cols.extend(portfolio_attributes)
    for val in genome.connections.values():
        print(X_cols[abs(val.key[0]) - 1], val.weight)

    # print log
    genome_portfolio.log.head(genome_portfolio.log.last_valid_index())

    # open spy
    with open("spy.pickle", "rb") as f:
        spy = pickle.load(f)

    first_close = spy.loc[spy.index.date == validation.index[0], "Close"]
    last_close = spy.loc[spy.index.date == validation.index[-1], "Close"]
    spy_return = last_close[0] / first_close[0]
    print("spy_return", spy_return)
    print("genome_portfolio_return", genome_portfolio.total_portfolio_value / cash)


if __name__ == "__main__":
    main()

