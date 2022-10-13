import gym
import math
import numpy as np

import pandas as pd
pd.options.mode.chained_assignment = None

from matplotlib import pyplot as plt


class TradingEnv(gym.Env):
    def __init__(
            self,
            symbol,
            date_start,
            date_end,
            portfolio,
            state_duration=30,
            slippage=0,
            episode_start_from=0
    ):

        csv_name = "".join(['Data/', symbol, '_', date_start, '_', date_end, '.csv'])
        self.data = pd.read_csv(csv_name, header=0, index_col='Timestamp', parse_dates=True)

        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(portfolio)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        self.state = [self.data['Close'][0:state_duration].tolist(),
                      self.data['Low'][0:state_duration].tolist(),
                      self.data['High'][0:state_duration].tolist(),
                      self.data['Volume'][0:state_duration].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0

        self.symbol = symbol
        self.date_start = date_start
        self.date_end = date_end
        self.state_duration = state_duration
        self.t = state_duration

        # max price deviation in [dt]
        self.cauchy = 0.1
        self.holding_amt = 0
        self.slippage = slippage

        if episode_start_from:
            self.setStartingPoint(episode_start_from)

    @property
    def position(self):
        return self.data['Position']

    @property
    def cash(self):
        return self.data['Cash']

    @property
    def holding_val(self):
        return self.data['Holdings']

    @property
    def close(self):
        return self.data['Close']

    @property
    def low(self):
        return self.data['Low']

    @property
    def high(self):
        return self.data['High']

    @property
    def volume(self):
        return self.data['Volume']

    @property
    def action(self):
        return self.data['Action']

    @property
    def money(self):
        return self.data['Money']

    @property
    def returns(self):
        return self.data['Returns']

    def reset(self, *args):
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Reset the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:self.state_duration].tolist(),
                      self.data['Low'][0:self.state_duration].tolist(),
                      self.data['High'][0:self.state_duration].tolist(),
                      self.data['Volume'][0:self.state_duration].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0

        # Reset additional variables related to the trading activity
        self.t = self.state_duration
        self.holding_amt = 0

        return self.state

    def _share_lower_bound(self, cash, holding_shares_amt, price):
        # please refer to the original paper
        nominator = - cash - holding_shares_amt * price * (1 + self.cauchy) * (1 + self.slippage)
        if nominator < 0:
            return nominator / (price * (2 * self.slippage + self.cauchy * (1 + self.slippage)))

        return nominator / (price * self.cauchy * (1 + self.slippage))

    def step(self, action):
        t = self.t
        numberOfShares = self.holding_amt

        # CASE 1: LONG POSITION
        if action == 1:
            self.position[t] = 1
            if self.position[t-1] == 1:
                self.cash[t] = self.cash[t-1]
                self.holding_val[t] = self.holding_amt * self.close[t]
            elif self.position[t-1] == 0:
                self.holding_amt = math.floor(self.cash[t-1] / (self.close[t] * (1. + self.slippage)))
                self.cash[t] = self.cash[t-1] - self.holding_amt * self.close[t] * (1. + self.slippage)
                self.holding_val[t] = self.holding_amt * self.close[t]
                self.action[t] = 1
            else:
                self.cash[t] = self.cash[t-1] - self.holding_amt * self.close[t] * (1. + self.slippage)
                self.holding_amt = math.floor(self.cash[t] / (self.close[t] * (1. + self.slippage)))
                self.cash[t] = self.cash[t] - self.holding_amt * self.close[t] * (1. + self.slippage)
                self.holding_val[t] = self.holding_amt * self.close[t]
                self.action[t] = 1

        # CASE 2: SHORT POSITION
        else:
            self.position[t] = -1
            # Case a: Short -> Short
            if self.position[t-1] == -1:
                lower_bound = self._share_lower_bound(self.cash[t-1], -self.holding_amt, self.close[t-1])
                if lower_bound <= 0:
                    self.cash[t] = self.cash[t-1]
                    self.holding_val[t] = -self.holding_amt * self.close[t]
                else:
                    shares2buy = min(math.floor(lower_bound), self.holding_amt)
                    self.holding_amt -= shares2buy
                    self.cash[t] = self.cash[t-1] - shares2buy * self.close[t] * (1 + self.slippage)
                    self.holding_val[t] = -self.holding_amt * self.close[t]

            elif self.position[t-1] == 0:
                self.holding_amt = math.floor(self.cash[t-1] / (self.close[t] * (1 + self.slippage)))
                self.cash[t] = self.cash[t-1] + self.holding_amt * self.close[t] * (1 - self.slippage)
                self.holding_val[t] = -self.holding_amt * self.close[t]
                self.action[t] = -1
            else:
                self.cash[t] = self.cash[t-1] + self.holding_amt * self.close[t] * (1 - self.slippage)
                self.holding_amt = math.floor(self.cash[t] / (self.close[t] * (1 + self.slippage)))
                self.cash[t] = self.cash[t] + self.holding_amt * self.close[t] * (1 - self.slippage)
                self.holding_val[t] = -self.holding_amt * self.close[t]
                self.action[t] = -1

        self.money[t] = self.holding_val[t] + self.cash[t]
        self.returns[t] = (self.money[t] - self.money[t-1]) / self.money[t-1]

        self.reward = self.returns[t]

        self.t = self.t + 1
        self.state = [self.data['Close'][self.t - self.state_duration: self.t].tolist(),
                      self.data['Low'][self.t - self.state_duration: self.t].tolist(),
                      self.data['High'][self.t - self.state_duration: self.t].tolist(),
                      self.data['Volume'][self.t - self.state_duration: self.t].tolist(),
                      [self.data['Position'][self.t - 1]]]

        if self.t == self.data.shape[0]:
            self.done = 1  

        action0 = int(not bool(action))
        customReward = False

        if action0 == 1:
            position0 = 1
            if self.position[t-1] == 1:
                cash0 = self.cash[t - 1]
                holding_val0 = numberOfShares * self.close[t]
            elif self.position[t-1] == 0:
                numberOfShares = math.floor(self.cash[t-1] / (self.close[t] * (1 + self.slippage)))
                cash0 = self.cash[t-1] - numberOfShares * self.close[t] * (1 + self.slippage)
                holding_val0 = numberOfShares * self.close[t]
            else:
                cash0 = self.cash[t-1] - numberOfShares * self.close[t] * (1 + self.slippage)
                numberOfShares = math.floor(cash0 / (self.close[t] * (1 + self.slippage)))
                cash0 = cash0 - numberOfShares * self.close[t] * (1 + self.slippage)
                holding_val0 = numberOfShares * self.close[t]
        else:
            position0 = -1
            if self.position[t-1] == -1:
                lower_bound = self._share_lower_bound(self.cash[t-1], -numberOfShares, self.close[t-1])
                if lower_bound <= 0:
                    cash0 = self.cash[t-1]
                    holding_val0 = -numberOfShares * self.close[t]
                else:
                    shares2buy = min(math.floor(lower_bound), numberOfShares)
                    numberOfShares -= shares2buy
                    cash0 = self.cash[t-1] - numberOfShares * self.close[t] * (1 + self.slippage)
                    holding_val0 = -numberOfShares * self.close[t]
            elif self.position[t-1] == 0:
                numberOfShares = math.floor(self.cash[t-1] / (self.close[t] * (1 + self.slippage)))
                cash0 = self.cash[t-1] + numberOfShares * self.close[t] * (1 - self.slippage)
                holding_val0 = -numberOfShares * self.close[t]
            else:
                cash0 = self.cash[t-1] + numberOfShares * self.close[t] * (1 - self.slippage)
                numberOfShares = math.floor(cash0 / (self.close[t] * (1 + self.slippage)))
                cash0 = cash0 + numberOfShares * self.close[t] * (1 - self.slippage)
                holding_val0 = -self.holding_amt * self.close[t]

        otherMoney = holding_val0 + cash0
        if not customReward:
            otherReward = (otherMoney - self.data['Money'][t-1])/self.data['Money'][t-1]
        else:
            otherReward = (self.data['Close'][t-1] - self.data['Close'][t])/self.data['Close'][t-1]
        otherState = [self.data['Close'][self.t - self.state_duration: self.t].tolist(),
                      self.data['Low'][self.t - self.state_duration: self.t].tolist(),
                      self.data['High'][self.t - self.state_duration: self.t].tolist(),
                      self.data['Volume'][self.t - self.state_duration: self.t].tolist(),
                      [position0]]
        self.info = {'State': otherState, 'Reward': otherReward, 'Done': self.done}

        return self.state, self.reward, self.done, self.info

    def render(self):
        """
        GOAL: Illustrate graphically the trading activity, by plotting
              both the evolution of the stock market price and the 
              evolution of the trading capital. All the trading decisions
              (long and short positions) are displayed as well.
        
        INPUTS: /   
        
        OUTPUTS: /
        """

        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the stock market price
        self.data['Close'].plot(ax=ax1, color='blue', lw=2)
        ax1.plot(self.data.loc[self.data['Action'] == 1.0].index, 
                 self.data['Close'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax1.plot(self.data.loc[self.data['Action'] == -1.0].index, 
                 self.data['Close'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        
        # Plot the second graph -> Evolution of the trading capital
        self.data['Money'].plot(ax=ax2, color='blue', lw=2)
        ax2.plot(self.data.loc[self.data['Action'] == 1.0].index, 
                 self.data['Money'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax2.plot(self.data.loc[self.data['Action'] == -1.0].index, 
                 self.data['Money'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        
        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long",  "Short"])
        ax2.legend(["Capital", "Long", "Short"])
        plt.savefig(''.join(['Figures/', str(self.symbol), '_Rendering', '.png']))
        plt.show()

    def setStartingPoint(self, startingPoint):
        """
        GOAL: Setting an arbitrary starting point regarding the trading activity.
              This technique is used for better generalization of the RL agent.
        
        INPUTS: - startingPoint: Optional starting point (iteration) of the trading activity.
        
        OUTPUTS: /
        """

        # Setting a custom starting point
        self.t = np.clip(startingPoint, self.state_duration, len(self.data.index))

        # Set the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][self.t - self.state_duration: self.t].tolist(),
                      self.data['Low'][self.t - self.state_duration: self.t].tolist(),
                      self.data['High'][self.t - self.state_duration: self.t].tolist(),
                      self.data['Volume'][self.t - self.state_duration: self.t].tolist(),
                      [self.data['Position'][self.t - 1]]]
        if(self.t == self.data.shape[0]):
            self.done = 1


if __name__ == '__main__':
    date_start = '2012-1-1'
    date_split = '2018-1-1'
    date_end = '2020-1-1'
    env = TradingEnv('AAPL', date_start, date_split, 10000)
