"""Calculate and backtest a strategy quickly. Good for tuning a specific strategy such as technical_levels."""

import datetime
import numpy as np

def random_sample_stocks(stock_list, num_draws):
    if len(stock_list)<num_draws:
        raise ValueError("num_draws larger than stock_list")

    idx_lst = np.random.randint(0, len(stock_list)-1, num_draws)
    drawn_stocks = []
    for idx in idx_lst:
        drawn_stocks.append(stock_list[idx])

    return drawn_stocks

def get_visual_feedback(stock, images):

    fig, ax = plt.subplots(2, 4)
    fig.set_size_inches(10,8)
    fig.subplots_adjust(left=0, bottom=0, top=1, right=1)

    for i in range(len(ax)):
        for x in range(len(ax[0])):
            ax[i][x].axis('off')
            ax[i][x].set_title(image_names[i*4+x])
            ax[i][x].imshow(images[i*4+x])
    plt.ion()
    plt.show()
    plt.pause(0.001)
    user_input = str(input(f'Is this a good parameter combo: y/n'))
    plt.close()
    return user_input








