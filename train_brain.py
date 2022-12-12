#!/usr/bin/env python
# coding: utf-8

from stock_class import Stock
import datetime
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import forecast
import brain
import random
import os


def main(num_workers=8, max_gen=50, restore_checkpoint=False, mini_batch_size=50, last_days_only=None):
    """train the NEAT algorithm and save checkpoints
    max_gen: number of generations to train
    restore_checkpoint_name: bool, if True searches for the latest saved checkpoint to continue training
    mini_batch_size: int or None, how many days to select randomly for training one generation, if None use whole dataset
    last_days_only: int or None, if int then brain_df.loc[brain_df.index.unique()[-last_days_only:], :]
        is used to reduce the size of the df for testing, if None then the whole brain_df is used which takes a lot longer
    """
    # open data for NEAT
    with open('brain_df.pickle', 'rb') as f:
        total_df = pickle.load(f)

    if mini_batch_size:
        if last_days_only:
            raise ValueError("One of 'mini_batch_size' or 'last_days_only' should be None")

    if last_days_only:
        last_days = total_df.index.unique()[-last_days_only:]
        brain_df = total_df.loc[last_days, :]

    if mini_batch_size:
        unique_dates = list(total_df.index.unique())

    # loop generation by generation
    count_gen = 0
    run = True
    while run:
        # select random mini-batch for every single generation
        if mini_batch_size:
            random_start_index = random.randint(0, len(unique_dates) - mini_batch_size)
            end_index = random_start_index + mini_batch_size
            selected_dates = unique_dates[random_start_index:end_index]
            brain_df = total_df.loc[selected_dates, :].copy()
            print("-------------------------------------------")
            print(f"generation {count_gen} -- Date: {unique_dates[random_start_index]}")

        # get last checkpoint
        if count_gen == 0:
            if restore_checkpoint:
                files = os.listdir()
                checkpoints = [filename for filename in files if filename.startswith("neat-checkpoint")]
                spl = [item.split("-") for item in checkpoints]
                # last_checkpoint
                last_checkpoint = "neat-checkpoint-" + str(max([int(item[2]) for item in spl]))
                print(f"Latest checkpoint: {last_checkpoint}")
                # max_gen_temp and gen_interval_temp (for correctly picking up and saving)
                max_gen_temp = 1
                gen_interval_temp = 1
            else:
                last_checkpoint = None
                print("Latest checkpoint: None")
                max_gen_temp = 1
                gen_interval_temp = 1

        elif count_gen > 0:
            files = os.listdir()
            checkpoints = [filename for filename in files if filename.startswith("neat-checkpoint")]
            spl = [item.split("-") for item in checkpoints]
            last_checkpoint = "neat-checkpoint-" + str(max([int(item[2]) for item in spl]))
            print(f"Latest checkpoint: {last_checkpoint}")
            max_gen_temp = 3
            gen_interval_temp = 2

        # fit NEAT
        brain_df.sort_index(inplace=True)
        winner, winner_portfolio = brain.run_neat(config_file=r'neat_config.txt',
                                                  num_workers=num_workers,
                                                  max_gen=max_gen_temp,
                                                  cash=1000,
                                                  restore_checkpoint_name=last_checkpoint,
                                                  generation_interval=gen_interval_temp,
                                                  total_df=brain_df,
                                                  X_cols=['cat_1', 'cat_2', 'cat_3', 'Basic Materials',
                                                          'Communication Services',
                                                          'Consumer Cyclical', 'Consumer Defensive', 'Energy',
                                                          'Financial Services', 'Healthcare', 'Industrials',
                                                          'Real Estate',
                                                          'Technology', 'Utilities', 'variance_100', 'variance_global'],
                                                  portfolio_attributes=["proportion_invested", "entropy_stock",
                                                                        "entropy_sector"],
                                                  verbose=False)

        count_gen += 1
        if count_gen < max_gen:
            run = True
        else:
            run = False
    # plot winner_portfolio.balance
    # plt.pie(winner_portfolio.balance.groupby('sector')['value'].sum(),
    #         labels=winner_portfolio.balance.groupby('sector')['sector'].apply(set))

    # log - balance
    # winner_portfolio.log.head(30)
    # winner_portfolio.balance.head(20)


if __name__ == '__main__':
    main()
