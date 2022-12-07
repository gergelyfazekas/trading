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
import random
import time
# from neat.parallel import ParallelEvaluator




from pathos.multiprocessing import ProcessingPool as Pool


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None, maxtasksperchild=None):
        """
        eval_function should take one argument, a tuple of (genome object, config object),
        and return a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.maxtasksperchild = maxtasksperchild
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(processes=num_workers, maxtasksperchild=maxtasksperchild)

    def __del__(self):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()


    def evaluate(self, genomes, config):
        self.pool.restart()
        genomes_without_id = [genome for genome_id, genome in genomes]
        config_list = [config] * len(genomes_without_id)
        genome_config_tuples = list(zip(genomes_without_id, config_list))

        jobs = self.pool.amap(self.eval_function, genome_config_tuples)
        fitnesses = jobs.get()
        # print('fitnesses', '\n', fitnesses)
        for fit, genome in zip(fitnesses, genomes_without_id):
            genome.fitness = fit






def shape_config_file(config_file, X_cols, portfolio_attributes):
    """rewrites the config file of the NEAT algorithm to be consistent with the current number of inputs
    Should only be used inside run_neat
    """
    config_parser = configparser.ConfigParser()
    config_parser.read('neat_config.txt')
    len_X = len(X_cols)
    len_portf = len(portfolio_attributes)
    config_parser['DefaultGenome']['num_inputs'] = str((len_X + len_portf))
    with open(config_file, 'w') as c_file:
        config_parser.write(c_file)


def run_neat(config_file, total_df, num_workers=4, restore_checkpoint_name=None, generation_interval=5,
             time_seconds_interval=None,
             max_gen=300, X_cols=['forecast', 'sector_encoded', 'variance_100', 'variance_global'],
             portfolio_attributes=["proportion_invested"],
             cash=500, threshold=0.05, verbose=False):
    """args:
    portfolio_attributes: these should be functions that can be called without an argument, see total_portfolio_value"""

    shape_config_file(config_file, X_cols, portfolio_attributes)

    training_data = total_df.copy()
    X_cols = [col.lower() for col in X_cols]

    # multiprocessing separates portfolios to be in their own single-element portfolio_list
    # to have a final portfolio list we have to collect the single-element portfolio lists in a list
    portfolio_list_parallel = []

    # print(training_data)

    Stock.clear_stock_list()
    Stock.set_each(training_data)
    random.shuffle(Stock.stock_list)

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
        Stock.clear_stock_list()
        Stock.set_each(training_data)

        for genome_id, genome in genomes:
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
                        continue
                    output = net.activate(list(df))
                    decision, proportion = output_to_decision(output[0], threshold=threshold)

                    if decision == 'buy':
                        # amount is the number of stocks to buy, value is the amount * price
                        amount = (proportion * genome_portfolio.cash_current) / \
                                 stock.data.loc[current_date, "close"]
                        genome_portfolio.buy(stock=stock, amount=amount, as_of=current_date)
                    elif decision == 'sell':
                        # amount is the number of stocks to sell (negative if sell), value is the amount * price
                        # amount = -1 * ((proportion * genome_portfolio.get_stock_amount(stock)) / \
                        #                training_data[training_data['ticker'] == stock.name].loc[current_date, "close"])
                        amount = -1 * (proportion * genome_portfolio.get_stock_amount(stock))
                        genome_portfolio.sell(stock=stock, amount=amount, as_of=current_date)
                    elif not decision:
                        # do nothing within [0.5-threshold , 0.5+threshold]
                        pass
                    else:
                        raise NotImplementedError
                # End-of-day update
                genome_portfolio.update_balance_eod(as_of=current_date)

                # return for that day, this incentivizes the portfolio to always have a positive daily return
                # (last tuple, 1st element of that tuple), -1 to center, *200 to scale up
                # a 1.01 return equals a 0.01*200 = 2 fitness score (this is a good magnitude for the entropy fitnesses)
                genome.fitness += (genome_portfolio.portfolio_return_hist[-1][1] - 1) * 200

            # Fitness update daily so that the diversification is always held high
            # entropy_sector() is maximum 2.3 since we have fix 10 sectors
            # entropy_stock() can be large so we have to cap it not to suppress the total_value in the fitness
            # 2.3 can be achieved with a uniformly weighted 10 stock portfolio, above that we don't differentiate
            # a well diversified portfolio earns a 4 * (2.3 + 2.3) = 19 fitness which is comparable to 10 days of 1.01 return
            genome.fitness += 4 * (min(genome_portfolio.entropy_stock, 2.3) + genome_portfolio.entropy_sector)

            # Fitness calculation: total_value + number of trades conducted + daily entropy
            # genome_portfolio.update_total_portfolio_value(as_of=current_date)
            # if genome_portfolio.log.last_valid_index():
            #     genome.fitness = genome_portfolio.total_portfolio_value + genome_portfolio.log.last_valid_index()
            # else:
            #     genome.fitness = genome_portfolio.total_portfolio_value
            if verbose:
                print(current_date)
                print('genome fitness', genome.fitness)
                print('cash_current', genome_portfolio.cash_current)
                print('log', genome_portfolio.log.head(10))
                print('balance', genome_portfolio.balance)



    def eval_genomes_parallel(genome_config_tuple):
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

        genome = genome_config_tuple[0]
        config = genome_config_tuple[1]
        genome.fitness = 0
        genome_portfolio = Portfolio(genome=genome, cash=cash)
        portfolio_list_parallel.append(genome_portfolio)
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        for current_date in training_data.index.unique():
            for stock in Stock.stock_list:
                try:
                    df = stock.data.loc[current_date, X_cols]
                    for item in portfolio_attributes:
                        df[item] = genome_portfolio.__getattribute__(item)
                except KeyError:
                    continue
                output = net.activate(list(df))
                decision, proportion = output_to_decision(output[0], threshold=threshold)

                if decision == 'buy':
                    # amount is the number of stocks to buy, value is the amount * price
                    amount = (proportion * genome_portfolio.cash_current) / \
                             stock.data.loc[current_date, "close"]
                    genome_portfolio.buy(stock=stock, amount=amount, as_of=current_date)
                elif decision == 'sell':
                    # amount is the number of stocks to sell (negative if sell), value is the amount * price
                    # amount = -1 * ((proportion * genome_portfolio.get_stock_amount(stock)) / \
                    #                training_data[training_data['ticker'] == stock.name].loc[current_date, "close"])
                    amount = -1 * (proportion * genome_portfolio.get_stock_amount(stock))
                    genome_portfolio.sell(stock=stock, amount=amount, as_of=current_date)
                elif not decision:
                    # do nothing within [0.5-threshold , 0.5+threshold]
                    pass
                else:
                    raise NotImplementedError
            # End-of-day update
            genome_portfolio.update_balance_eod(as_of=current_date)

            # return for that day, this incentivizes the portfolio to always have a positive daily return
            # (last tuple, 1st element of that tuple), -1 to center, *200 to scale up
            # a 1.01 return equals a 0.01*200 = 2 fitness score (this is a good magnitude for the entropy fitnesses)
            genome.fitness += (genome_portfolio.portfolio_return_hist[-1][1] - 1) * 200

            # Fitness update daily so that the diversification is always held high
            # entropy_sector() is maximum 2.3 since we have fix 10 sectors
            # entropy_stock() can be large so we have to cap it not to suppress the total_value in the fitness
            # 2.3 can be achieved with a uniformly weighted 10 stock portfolio, above that we don't differentiate
            # a well diversified portfolio earns a 4 * (2.3 + 2.3) = 19 fitness which is comparable to 10 days of 1.01 return
        genome.fitness += 4 * (min(genome_portfolio.entropy_stock, 2.3) + genome_portfolio.entropy_sector)

        # Fitness calculation: total_value + number of trades conducted + daily entropy
        # genome_portfolio.update_total_portfolio_value(as_of=current_date)
        # if genome_portfolio.log.last_valid_index():
        #     genome.fitness = genome_portfolio.total_portfolio_value + genome_portfolio.log.last_valid_index()
        # else:
        #     genome.fitness = genome_portfolio.total_portfolio_value
        if verbose:
            print(current_date)
            print('genome fitness', genome.fitness)
            print('cash_current', genome_portfolio.cash_current)
            print('log', genome_portfolio.log.head(10))
            print('balance', genome_portfolio.balance)
        # end of date loop
        return genome.fitness

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    if restore_checkpoint_name:
        p = neat.Checkpointer.restore_checkpoint(restore_checkpoint_name)
    else:
        p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(generation_interval=generation_interval, time_interval_seconds=time_seconds_interval))

    # Run
    # Single CPU
    if num_workers == 1:
        winner = p.run(eval_genomes, max_gen)
        # get winner portfolio
        winner_portf = [portf for portf in Portfolio.portfolio_list if portf.genome == winner][0]
    # Multiple CPU
    elif num_workers > 1:
        par_eval = ParallelEvaluator(num_workers=num_workers, eval_function=eval_genomes_parallel)
        # winner = p.run(ParallelEvaluator(num_workers=num_workers, eval_function=eval_genomes_parallel).evaluate, max_gen)
        winner = p.run(par_eval.evaluate, max_gen)
        par_eval.__del__()

        # can't get winner portfolio yet --- TO DO
        try:
            winner_portf = [portf for portf in portfolio_list_parallel if portf.genome == winner][0]
        except IndexError:
            winner_portf = None
            # print('Unable to return winner portfolio yet')
    else:
        raise ValueError("Invalid num_workers")

    # pickle the winner
    with open("best_genome.pickle", "wb") as f:
        pickle.dump(winner, f)

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
