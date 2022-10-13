# coding=utf-8

import math
import random
import copy
import datetime

import numpy as np

from collections import deque
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tradingPerformance import PerformanceEstimator
from dataAugmentation import DataAugmentation
from tradingEnv import TradingEnv

random.seed(0xCAFFE)
torch.manual_seed(0xCAFFE)


class ReplayMemory:

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batchSize):
        state, action, reward, next_state, done = zip(*random.sample(self.memory, batchSize))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = deque(maxlen=self.capacity)


class DQN(nn.Module):
    """
    x -> fc1() -> batchNormalization1() -> leakyRelu() -> dropOut1()
      -> fc2() -> batchNormalization2() -> leakyRelu() -> dropOut2()
      -> fc3() -> batchNormalization3() -> leakyRelu() -> dropOut3()
      -> fc4() -> batchNormalization4() -> leakyRelu() -> dropOut4()
      -> fc5() -> OUTPUT
    """

    def __init__(
            self,
            nin,
            nout,
            neurons=512,
            dropout=0.2
    ):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(nin, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, neurons)
        self.fc4 = nn.Linear(neurons, neurons)
        self.fc5 = nn.Linear(neurons, nout)

        self.bn1 = nn.BatchNorm1d(neurons)
        self.bn2 = nn.BatchNorm1d(neurons)
        self.bn3 = nn.BatchNorm1d(neurons)
        self.bn4 = nn.BatchNorm1d(neurons)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)

    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        return x


class TDQN:
    def __init__(
            self,
            n_obs,
            n_act,
            neurons=512,
            dropout=0.2,
            eps_start=1.,
            eps_end=0.01,
            eps_decay=10000,
    ):

        random.seed(0)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # reward discount factor
        self.gamma = 0.4
        self.learning_rate = 0.0001
        self.l2_factor = 0.000001
        self.target_network_update_interval = 1000
        self.sticky_action_probability = 0.1
        self.gradient_clip = 1
        self.episode = 5

        self.low_pass_filter_order = 5

        # batch size to collect data from replay buffer
        self.bs = 32
        self.replay_memory = ReplayMemory(100000)

        self.n_obs = n_obs
        self.n_act = n_act

        self.policy_network = DQN(n_obs, n_act, neurons, dropout).to(self.device)
        self.target_network = DQN(n_obs, n_act, neurons, dropout).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.policy_network.eval()
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_factor
        )
        self.epsilon = lambda iteration: eps_end + (eps_start - eps_end) * math.exp(-1 * iteration / eps_decay)
        self.iterations = 0
        self.writer = SummaryWriter('runs/' + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S"))

    @staticmethod
    def get_features_stats(training_env):
        d = training_env.data
        c = d['Close'].tolist()
        l = d['Low'].tolist()
        h = d['High'].tolist()
        v = d['Volume'].tolist()

        stats = []
        margin = 1

        rtns = [abs((c[i] - c[i-1]) / c[i-1]) for i in range(1, len(c))]
        stats.append((0, np.max(rtns) * margin))

        price_ranges = [abs(h[i] - l[i]) for i in range(len(l))]
        stats.append((0, np.max(price_ranges) * margin))

        stats.append((0, 1))
        stats.append((np.min(v) / margin, np.max(v) * margin))
        
        return stats

    @staticmethod
    def preprocess_state(s, stats):
        """
        state return from trading_env is in the form:
            [[Close, ], [Low, ], [High, ], [Vol, ], [Last_Position, ]]
            if state time duration is 30 for instance, the length of the state been
            flatten is (30 - 1) * 4 + 1 = 117
        """

        # close, low, high, vol
        c, l, h, v, last_position = s[0], s[1], s[2], s[3], s[4]
        features = []

        # normalize return in [dt] to [0, 1]
        # rtn_min set to 0, the statistical value of rtn is calculated after ABS op
        rtn_min, rtn_max = stats[0][0], stats[0][1]

        rtns = [(c[i]-c[i-1]) / c[i-1] for i in range(1, len(c))]
        if rtn_min != rtn_max:
            features.extend([((x - rtn_min) / (rtn_max - rtn_min)) for x in rtns])
        else:
            features.extend([0 for _ in rtns])

        # normalize price volatility in [dt] to [0, 1]
        pr_min, pr_max = stats[1][0], stats[1][1]
        price_ranges = [abs(h[i] - l[i]) for i in range(1, len(l))]

        if pr_min != pr_max:
            features.extend([((x - pr_min) / (pr_max - pr_min)) for x in price_ranges])
        else:
            features.extend([0 for _ in price_ranges])

        # merge c, l, h to a new feature, close price's relative postion
        features.extend([abs(c[i] - l[i]) / abs(h[i] - l[i]) if abs(h[i] - l[i]) != 0 else 0.5
                         for i in range(1, len(c))])

        # normalize trading volume in [dt] to [0, 1]
        vol_min, vol_max = stats[3][0], stats[3][1]

        if vol_min != vol_max:
            features.extend([((x - vol_min) / (vol_max - vol_min)) for x in v[1:]])
        else:
            features.extend([0 for _ in v[1:]])

        features.extend(last_position)

        return features

    @staticmethod
    def clip_reward(reward):
        # clip the reward to the range [-1, 1]
        return np.clip(reward, -1, 1)

    def update_target_network(self):
        if self.iterations % self.target_network_update_interval == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def greedy_action(self, state):
        with torch.no_grad():
            # forward the state with policy network to get the Q values
            state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            qs = self.policy_network(state_tensor).squeeze(0)

            q, action = qs.max(0)

            return action.item(), q.item(), qs.cpu().numpy()

    def epsilon_greedy_action(self, state, previous_action):
        self.iterations += 1

        if random.random() > self.epsilon(self.iterations - 1):
            if random.random() > self.sticky_action_probability:
                return self.greedy_action(state)

            return previous_action, 0, [0, 0]

        return random.randrange(self.n_act), 0, [0, 0]

    def learning(self):
        if len(self.replay_memory) < self.bs:
            return

        self.policy_network.train()

        state, action, reward, next_state, done = self.replay_memory.sample(self.bs)
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.float, device=self.device)

        q = self.policy_network(state).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_action = torch.max(self.policy_network(next_state), 1)[1]
            next_q = self.target_network(next_state).gather(1, next_action.unsqueeze(1)).squeeze(1)
            target_q = reward + self.gamma * next_q * (1 - done)

        # Huber loss
        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.gradient_clip)

        self.optimizer.step()
        self.update_target_network()
        self.policy_network.eval()

    def training(self, training_env):
        ds = DataAugmentation()
        training_envs = ds.generate(training_env)

        score = np.zeros((len(training_envs), self.episode))

        testing_env = TradingEnv(
            training_env.symbol,
            training_env.date_end,
            '2020-1-1',
            training_env.data['Money'][0],
            training_env.state_duration,
            training_env.slippage
        )

        train_sharpe_list = []
        test_sharpe_list = []

        print("Training progression (hardware selected => " + str(self.device) + "):")

        for ieps in tqdm(range(self.episode)):
            for i, env in enumerate(training_envs):
                stats = self.get_features_stats(env)
                env.reset()
                env.setStartingPoint(random.randrange(len(env.data)))

                state = self.preprocess_state(env.state, stats)
                previous_action, done, steps_counter = 0, 0, 0

                total_reward = 0

                while not done:
                    action, _, _ = self.epsilon_greedy_action(state, previous_action)
                    next_state, reward, done, info = env.step(action)

                    reward = self.clip_reward(reward)
                    next_state = self.preprocess_state(next_state, stats)
                    self.replay_memory.push(state, action, reward, next_state, done)

                    # Trick for better exploration in TDQN paper
                    other_action = int(not bool(action))
                    other_reward = self.clip_reward(info['Reward'])
                    other_next_state = self.preprocess_state(info['State'], stats)
                    other_done = info['Done']
                    self.replay_memory.push(state, other_action, other_reward, other_next_state, other_done)

                    self.learning()

                    # Update the RL state
                    state = next_state
                    previous_action = action

                    total_reward += reward

                score[i][ieps] = total_reward

            training_env = self.testing(training_env, training_env)
            train_sharpe = PerformanceEstimator(training_env.data).computeSharpeRatio()
            train_sharpe_list.append(train_sharpe)
            self.writer.add_scalar('train_sharpe_ratio', train_sharpe, ieps)
            training_env.reset()

            testing_env = self.testing(training_env, testing_env)
            test_sharpe = PerformanceEstimator(testing_env.data).computeSharpeRatio()
            test_sharpe_list.append(test_sharpe)
            self.writer.add_scalar('test_sharpe_ratio', test_sharpe, ieps)
            testing_env.reset()
        
        training_env = self.testing(training_env, training_env)
        # training_env.render()

        self.plot_train_test_performance(
            train_sharpe_list,
            test_sharpe_list,
            training_env.symbol,
            'train_test_sharpe_ratio'
        )
        # for i in range(len(training_envs)):
        #     self.plot_training_total_reward(score[i], training_env.marketSymbol)

        analyser = PerformanceEstimator(training_env.data)
        analyser.displayPerformance('TDQN')
        
        self.writer.close()

        return training_env

    def testing(self, training_env, env, verbose=False):
        ds = DataAugmentation()
        env_smoothed = ds.lowPassFilter(env, self.low_pass_filter_order)
        training_env = ds.lowPassFilter(training_env, self.low_pass_filter_order)

        stats = self.get_features_stats(training_env)
        state = self.preprocess_state(env_smoothed.reset(), stats)
        env.reset()

        qs0, qs1 = [], []
        done = 0

        while not done:
            action, _, q = self.greedy_action(state)
                
            next_state, _, done, _ = env_smoothed.step(action)
            env.step(action)
                
            state = self.preprocess_state(next_state, stats)

            qs0.append(q[0])
            qs1.append(q[1])

        if verbose:
            # env.render()
            # self.plotQValues(qs0, qs1, env.marketSymbol)

            analyser = PerformanceEstimator(env.data)
            analyser.displayPerformance('TDQN')
        
        return env

    @staticmethod
    def plot_training_total_reward(score, symbol):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Total Reward', xlabel='Episode')
        ax1.plot(score)
        plt.savefig(''.join(['Figures/', str(symbol), '_TrainingResults', '.png']))
        #plt.show()

    @staticmethod
    def plot_train_test_performance(train_list, test_list, symbol, title):
        fig = plt.figure()
        ax = fig.add_subplot(111, ylabel=title, xlabel='Episode')
        ax.plot(train_list)
        ax.plot(test_list)
        ax.legend(["Training", "Testing"])
        plt.savefig(''.join(['Figures/', symbol, '_', title, '.png']))
        plt.show()

    def plotQValues(self, QValues0, QValues1, marketSymbol):
        """
        Plot sequentially the Q values related to both actions.
        
        :param: - QValues0: Array of Q values linked to action 0.
                - QValues1: Array of Q values linked to action 1.
                - marketSymbol: Stock market trading symbol.
        
        :return: /
        """

        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Q values', xlabel='Time')
        ax1.plot(QValues0)
        ax1.plot(QValues1)
        ax1.legend(['Short', 'Long'])
        plt.savefig(''.join(['Figures/', str(marketSymbol), '_QValues', '.png']))
        #plt.show()

    def saveModel(self, fileName):
        torch.save(self.policy_network.state_dict(), fileName)

    def loadModel(self, fileName):
        self.policy_network.load_state_dict(torch.load(fileName, map_location=self.device))
        self.target_network.load_state_dict(self.policy_network.state_dict())
