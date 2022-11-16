"""For a given day takes a look at one stock and decides to buy, accumulate, decrease, sell, do nothing.

Also the training of this genetic algorithm takes place here.
"""
import datetime
import pickle
from stock_class import Stock
import neat
import os
from portfolio_class import Portfolio
import pandas as pd
import database
import configparser


def shape_config_file(config_file, X_cols):
    """rewrites the config file of the NEAT algorithm to be consistent with the current number of inputs
    Should only be used inside run_neat
    """
    config_parser = configparser.ConfigParser()
    config_parser.read('neat_config.txt')
    config_parser['DefaultGenome']['num_inputs'] = str(len(X_cols))
    with open(config_file, 'w') as c_file:
        config_parser.write(c_file)


def run_neat(config_file, pickeled_df, max_gen=300,
             X_cols = ['forecast', 'sector_encoded', 'variance_100', 'variance_global'],
             cash=500, stock_names=None, threshold=0.05):

    shape_config_file(config_file, X_cols)

    training_data = pickeled_df.copy()
    # print(training_data)

    # Stock.clear_stock_list()

    def eval_genomes(genomes, config):
        """evaluates every portfolio's genome by looping through
            - all the dates of the training set and
            - within every day it loops through all the stocks and
            - decides what the action is and adjusts (buy/sell) the portfolio
            - at the last date takes a look at the portfolio value to determine fitness

            parameters:
            genomes, config -- neat algorithm parameters
            stocks -- list of stock instances to choose from
            training_data -- a pd.df in sql format such as:
                                     - the index is a datetime.date object and can be filtered by that
                                     - all columns are covariates (no label/target is included)
            cash_current -- the dollar value of a portfolio's budget
            """

        # for ticker in training_data['ticker'].unique():
        #     if not stock_names:
        #         Stock(ticker)
        #     else:
        #         if ticker in stock_names:
        #             Stock(ticker)
        # for stock in Stock.stock_list:
        #     stock.set_data(training_data[training_data['ticker'] == stock.name])
        #     stock.lowercase()
        #     stock.set_index()

        if not Stock.stock_list:
            if not stock_names:
                Stock.set_each(training_data)

        # training_data.set_index("date_", inplace=True)
        for genome_id, genome in genomes:
            genome_portfolio = Portfolio(genome=genome, cash=cash)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for current_date in training_data.index.unique():
                # print('current_date', current_date)
                for stock in Stock.stock_list:
                    # print(stock.name)
                    df = training_data[training_data['ticker'] == stock.name].loc[
                        current_date, X_cols]
                    output = net.activate(list(df))
                    # print('net output', output)
                    decision, proportion = output_to_decision(output[0], threshold=threshold)

                    # currently the models proportion is applied on the remaining cash if buy so that we can never go beyond budget
                    # for sell it is the proportion of the current amount (count) of the stock in the portfolio
                    # another way would be to apply the proportion on the current portfolio value of the stock in question
                    # but then we would have to discourage the behaviour of going beyond budget by reducing the fitness funciton
                    # if genome_portfolio.got_enough_cash returns False
                    if decision == 'buy':
                        amount = (proportion * genome_portfolio.cash_current) / \
                                 training_data[training_data['ticker'] == stock.name].loc[current_date, "close"]
                        genome_portfolio.buy(stock=stock, amount=amount, as_of=current_date)
                    elif decision == 'sell':
                        amount = -1 * ((proportion * genome_portfolio.get_stock_amount(stock)) / \
                                       training_data[training_data['ticker'] == stock.name].loc[current_date, "close"])
                        genome_portfolio.sell(stock=stock, amount=amount, as_of=current_date)
                    elif not decision:
                        # do nothing within [0.5-threshold , 0.5+threshold]
                        pass
                    else:
                        raise NotImplementedError
            # fitness = portfolio value at the last date of the training set
            # other option would be to check the portfolios value at the end of every year:
            #   - and reduce the initial 500 with the place of the genome compared to the others
            #   - so if the genome is the best every year it gets 500 - (10 years * 1st place) = 490
            #   - if it is the 3rd every year then 500 - 10*3 = 470 ...
            #   - this fitness function would incentivize interim good performance
            #  We could also consider to terminate if genome.fitness reached a level
            genome_portfolio.update_total_portfolio_value(as_of=current_date)
            genome.fitness = genome_portfolio.total_portfolio_value
            # print('genome_portfolio', genome_portfolio.balance)
            print('log', genome_portfolio.log.head(10))


    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(1))

    # Run for up to max_gen generations.
    winner = p.run(eval_genomes, max_gen)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

    winner_portf = [portf for portf in Portfolio.portfolio_list if portf.genome == winner][0]
    return winner, winner_portf



def output_to_decision(output, threshold):
    """determine the direction (buy/sell) and
     the proportion which will be converted to an actual amount from a sigmoid (0-1 range) output"""
    if output < 0.5 - threshold:
        decision = 'sell'
    elif output > 0.5 + threshold:
        decision = 'buy'
    else:
        decision = None
    proportion = 2 * abs(output - 0.5)
    return decision, proportion


