"""For a given day takes a look at one stock and decides to buy, accumulate, decrease, sell, do nothing.

Also the training of this genetic algorithm takes place here.
"""


import neat
import os
from portfolio_class import Portfolio


def run_neat(config_file, max_gen=300):
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
    p.add_reporter(neat.Checkpointer(1))

    # Run for up to max_gen generations.
    winner = p.run(eval_genomes, max_gen)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

    winner_portfolio_value = [portf.total_portfolio_value for portf in Portfolio.portfolio_list if portf.genome == winner]
    print("winner genome", winner, "final portfolio value", winner_portfolio_value)


def eval_genomes(genomes, config, stocks, training_data, cash = 500):
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
        cash -- the dollar value of a portfolio's budget
        """
    for genome_id, genome in genomes:
        genome_portfolio = Portfolio(genome = genome, cash = cash)
        for current_date in training_data.index:
            for stock in stocks:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                output = net.activate(stock.data.iloc[:, 1:])
                decision, proportion = output_to_decision(output, threshold=0.05)

                # currently the models proportion is applied on the remaining cash if buy so that we can never go beyond budget
                # for sell it is the proportion of the current amount (count) of the stock in the portfolio
                # another way would be to apply the proportion on the current portfolio value of the stock in question
                # but then we would have to discourage the behaviour of going beyond budget by reducing the fitness funciton
                # if genome_portfolio.got_enough_cash returns False


                if decision == 'buy':
                    amount = (proportion * genome_portfolio.cash) / stock.get_price(current_date)
                    genome_portfolio.buy(stock = stock, amount = amount, date = current_date)
                elif decision == 'sell':
                    amount = (proportion * genome_portfolio.get_stock_amount(stock)) / stock.get_price(current_date)
                    genome_portfolio.sell(stock = stock, amount = amount, date = current_date)
                elif not decision:
                    # do nothing within [0.5-threshold , 0.5+threshold]
                    pass
        # fitness = portfolio value at the last date of the training set
        # other option would be to check the portfolios value at the end of every year:
        #   - and reduce the initial 500 with the place of the genome compared to the others
        #   - so if the genome is the best every year it gets 500 - (10 years * 1st place) = 490
        #   - if it is the 3rd every year then 500 - 10*3 = 470 ...
        #   - this fitness function would incentivize interim good performance
        #  We could also consider to terminate if genome.fitness reached a level
        genome.fitness = genome_portfolio.current_portfolio_value


def output_to_decision(output, threshold):
    """determine the direction (buy/sell) and
     the proportion which will be converted to an actual amount from a sigmoid (0-1 range) output"""
    if output < 0.5 - threshold:
        decision = 'sell'
    elif output > 0.5 + threshold:
        decision = 'buy'
    else:
        decision = None
    proportion = abs(output - 0.5)
    return decision, proportion

for day in days:
    for stock in stock_list:
        eval_genomes(genomes, config, stock)