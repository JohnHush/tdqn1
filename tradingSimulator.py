# coding=utf-8

import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from tradingEnv import TradingEnv
from TDQN import TDQN


def plotEntireTrading(trainingEnv, testingEnv, splitingDate):

    # Artificial trick to assert the continuity of the Money curve
    ratio = trainingEnv.data['Money'][-1]/testingEnv.data['Money'][0]
    testingEnv.data['Money'] = ratio * testingEnv.data['Money']

    # Concatenation of the training and testing trading dataframes
    dataframes = [trainingEnv.data, testingEnv.data]
    data = pd.concat(dataframes)

    # Set the Matplotlib figure and subplots
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
    ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

    # Plot the first graph -> Evolution of the stock market price
    trainingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2)
    testingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2, label='_nolegend_')
    ax1.plot(data.loc[data['Action'] == 1.0].index,
             data['Close'][data['Action'] == 1.0],
             '^', markersize=5, color='green')
    ax1.plot(data.loc[data['Action'] == -1.0].index,
             data['Close'][data['Action'] == -1.0],
             'v', markersize=5, color='red')

    # Plot the second graph -> Evolution of the trading capital
    trainingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2)
    testingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2, label='_nolegend_')
    ax2.plot(data.loc[data['Action'] == 1.0].index,
             data['Money'][data['Action'] == 1.0],
             '^', markersize=5, color='green')
    ax2.plot(data.loc[data['Action'] == -1.0].index,
             data['Money'][data['Action'] == -1.0],
             'v', markersize=5, color='red')

    # Plot the vertical line seperating the training and testing datasets
    ax1.axvline(pd.Timestamp(splitingDate), color='black', linewidth=2.0)
    ax2.axvline(pd.Timestamp(splitingDate), color='black', linewidth=2.0)

    # Generation of the two legends and plotting
    ax1.legend(["Price", "Long",  "Short", "Train/Test separation"])
    ax2.legend(["Capital", "Long", "Short", "Train/Test separation"])
    plt.savefig(''.join(['Figures/', str(trainingEnv.symbol), '_TrainingTestingRendering', '.png']))
    #plt.show()


def train_tqdn():
    symbol = 'AAPL'

    date_start = '2012-1-1'
    date_split = '2018-1-1'
    date_end = '2020-1-1'

    n_state = 30
    n_obs = 1 + (n_state - 1) * 4
    n_act = 2

    slippage = 0.001
    portfolio = 100000

    tdqn = TDQN(n_obs, n_act)

    training_env = TradingEnv(symbol, date_start, date_split, portfolio, n_state, slippage)
    training_env = tdqn.training(training_env)

    testing_env = TradingEnv(symbol, date_split, date_end, portfolio, n_state, slippage)
    testing_env = tdqn.testing(training_env, testing_env, verbose=True)

    # plotEntireTrading(training_env, testing_env, date_split)

    # return tdqn, training_env, testing_env


if __name__ == '__main__':
    train_tqdn()
