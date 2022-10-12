# coding=utf-8

import pandas as pd

from tqdm import tqdm
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from tradingEnv import TradingEnv
from TDQN import TDQN


# Variables defining the default trading horizon
startingDate = '2012-1-1'
endingDate = '2020-1-1'
splitingDate = '2018-1-1'

# Variables defining the default observation and state spaces
stateLength = 30
observationSpace = 1 + (stateLength-1)*4
actionSpace = 2

# Variables setting up the default transaction costs
percentageCosts = [0, 0.1, 0.2]
transactionCosts = percentageCosts[1]/100

# Variables specifying the default capital at the disposal of the trader
money = 100000

# Variables specifying the default general training parameters
numberOfEpisodes = 50


def plotEntireTrading(trainingEnv, testingEnv):

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
    plt.savefig(''.join(['Figures/', str(trainingEnv.marketSymbol), '_TrainingTestingRendering', '.png']))
    #plt.show()


def simulateNewStrategy():
    from TDQN import TDQN
    stock = 'AAPL'
    tradingStrategy = TDQN(observationSpace, actionSpace)
    trainingParameters = [numberOfEpisodes]

    verbose = True
    plotTraining = True
    rendering = True
    showPerformance = True

    trainingEnv = TradingEnv(stock, startingDate, splitingDate, money, stateLength, transactionCosts)

    # Training of the trading strategy
    trainingEnv = tradingStrategy.training(trainingEnv, trainingParameters=trainingParameters,
                                           verbose=verbose, rendering=rendering,
                                           plotTraining=plotTraining, showPerformance=showPerformance)

    # Initialize the trading environment associated with the testing phase
    testingEnv = TradingEnv(stock, splitingDate, endingDate, money, stateLength, transactionCosts)

    # Testing of the trading strategy
    testingEnv = tradingStrategy.testing(trainingEnv, testingEnv, rendering=rendering, showPerformance=showPerformance)

    # Show the entire unified rendering of the training and testing phases
    if rendering:
        plotEntireTrading(trainingEnv, testingEnv)

    return tradingStrategy, trainingEnv, testingEnv


if __name__ == '__main__':
    simulateNewStrategy()
