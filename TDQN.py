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
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tradingPerformance import PerformanceEstimator
from dataAugmentation import DataAugmentation
from tradingEnv import TradingEnv

random.seed(0xCAFFE)
torch.manual_seed(0xCAFFE)


# Default parameters related to the DQN algorithm
gamma = 0.4
learningRate = 0.0001
learningUpdatePeriod = 1

# Default parameters related to the Experience Replay mechanism
capacity = 100000
batchSize = 32
experiencesRequired = 1000

# Default parameters related to the Deep Neural Network
numberOfNeurons = 512
dropout = 0.2

# Default parameters related to the Epsilon-Greedy exploration technique
epsilonStart = 1.0
epsilonEnd = 0.01
epsilonDecay = 10000

# Default parameters related to preprocessing
filterOrder = 5

# Default paramters related to the clipping of both the gradient and the RL rewards
gradientClipping = 1

# Default parameter related to the L2 Regularization 
L2Factor = 0.000001

# Default paramter related to the hardware acceleration (CUDA)
GPUNumber = 0


class ReplayMemory:

    def __init__(self, capacity=capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def sample(self, batchSize):
        state, action, reward, nextState, done = zip(*random.sample(self.memory, batchSize))
        return state, action, reward, nextState, done

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = deque(maxlen=capacity)


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
    def __init__(self, observationSpace, actionSpace, numberOfNeurons=numberOfNeurons, dropout=dropout,
                 gamma=gamma, learningRate=learningRate,
                 epsilonStart=epsilonStart, epsilonEnd=epsilonEnd, epsilonDecay=epsilonDecay,
                 capacity=capacity, batchSize=batchSize):

        # Initialise the random function with a new random seed
        random.seed(0)

        # Check availability of CUDA for the hardware (CPU or GPU)
        self.device = torch.device('cuda:'+str(GPUNumber) if torch.cuda.is_available() else 'cpu')

        # Set the general parameters of the DQN algorithm
        self.gamma = gamma
        self.learningRate = learningRate

        self.target_network_update_interval = 1000
        self.sticky_action_probability = 0.1

        # Set the Experience Replay mechnism
        self.capacity = capacity
        self.batchSize = batchSize
        self.replayMemory = ReplayMemory(capacity)

        # Set both the observation and action spaces
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace

        # Set the two Deep Neural Networks of the DQN algorithm (policy and target)
        self.policyNetwork = DQN(observationSpace, actionSpace, numberOfNeurons, dropout).to(self.device)
        self.targetNetwork = DQN(observationSpace, actionSpace, numberOfNeurons, dropout).to(self.device)
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
        self.policyNetwork.eval()
        self.targetNetwork.eval()

        # Set the Deep Learning optimizer
        self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=learningRate, weight_decay=L2Factor)

        # Set the Epsilon-Greedy exploration technique
        self.epsilonValue = lambda iteration: epsilonEnd + (epsilonStart - epsilonEnd) * math.exp(-1 * iteration / epsilonDecay)
        
        # Initialization of the iterations counter
        self.iterations = 0

        # Initialization of the tensorboard writer
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
            self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())

    def greedy_action(self, state):
        with torch.no_grad():
            # forward the state with policy network to get the Q values
            state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            qs = self.policyNetwork(state_tensor).squeeze(0)

            q, action = qs.max(0)

            return action.item(), q.item(), qs.cpu().numpy()

    def epsilon_greedy_action(self, state, previous_action):
        self.iterations += 1

        if random.random() > self.epsilonValue(self.iterations - 1):
            if random.random() > self.sticky_action_probability:
                return self.greedy_action(state)

            return previous_action, 0, [0, 0]

        return random.randrange(self.actionSpace), 0, [0, 0]

    def learning(self, batchSize=batchSize):
        """
        GOAL: Sample a batch of past experiences and learn from it
              by updating the Reinforcement Learning policy.
        
        INPUTS: batchSize: Size of the batch to sample from the replay memory.
        
        OUTPUTS: /
        """
        
        # Check that the replay memory is filled enough
        if (len(self.replayMemory) >= batchSize):

            # Set the Deep Neural Network in training mode
            self.policyNetwork.train()

            # Sample a batch of experiences from the replay memory
            state, action, reward, nextState, done = self.replayMemory.sample(batchSize)

            # Initialization of Pytorch tensors for the RL experience elements
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            action = torch.tensor(action, dtype=torch.long, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float, device=self.device)
            nextState = torch.tensor(nextState, dtype=torch.float, device=self.device)
            done = torch.tensor(done, dtype=torch.float, device=self.device)

            # Compute the current Q values returned by the policy network
            currentQValues = self.policyNetwork(state).gather(1, action.unsqueeze(1)).squeeze(1)

            # Compute the next Q values returned by the target network
            with torch.no_grad():
                nextActions = torch.max(self.policyNetwork(nextState), 1)[1]
                nextQValues = self.targetNetwork(nextState).gather(1, nextActions.unsqueeze(1)).squeeze(1)
                expectedQValues = reward + gamma * nextQValues * (1 - done)

            # Compute the Huber loss
            loss = F.smooth_l1_loss(currentQValues, expectedQValues)

            # Computation of the gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), gradientClipping)

            # Perform the Deep Neural Network optimization
            self.optimizer.step()

            # If required, update the target deep neural network (update frequency)
            self.update_target_network()

            # Set back the Deep Neural Network in evaluation mode
            self.policyNetwork.eval()


    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        """
        GOAL: Train the RL trading agent by interacting with its trading environment.
        
        INPUTS: - trainingEnv: Training RL environment (known).
                - trainingParameters: Additional parameters associated
                                      with the training phase (e.g. the number
                                      of episodes).  
                - verbose: Enable the printing of a training feedback.
                - rendering: Enable the training environment rendering.
                - plotTraining: Enable the plotting of the training results.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - trainingEnv: Training RL environment.
        """

        """
        # Compute and plot the expected performance of the trading policy
        trainingEnv = self.plotExpectedPerformance(trainingEnv, trainingParameters, iterations=50)
        return trainingEnv
        """

        # Apply data augmentation techniques to improve the training set
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)

        # Initialization of some variables tracking the training and testing performances
        if plotTraining:
            # Training performance
            performanceTrain = []
            score = np.zeros((len(trainingEnvList), trainingParameters[0]))
            # Testing performance
            marketSymbol = trainingEnv.marketSymbol
            startingDate = trainingEnv.endingDate
            endingDate = '2020-1-1'
            money = trainingEnv.data['Money'][0]
            stateLength = trainingEnv.stateLength
            transactionCosts = trainingEnv.transactionCosts
            testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)
            performanceTest = []

        try:
            # If required, print the training progression
            if verbose:
                print("Training progression (hardware selected => " + str(self.device) + "):")

            # Training phase for the number of episodes specified as parameter
            for episode in tqdm(range(trainingParameters[0]), disable=not(verbose)):

                # For each episode, train on the entire set of training environments
                for i in range(len(trainingEnvList)):
                    
                    # Set the initial RL variables
                    coefficients = self.get_features_stats(trainingEnvList[i])
                    trainingEnvList[i].reset()
                    startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                    trainingEnvList[i].setStartingPoint(startingPoint)
                    state = self.preprocess_state(trainingEnvList[i].state, coefficients)
                    previousAction = 0
                    done = 0
                    stepsCounter = 0

                    # Set the performance tracking veriables
                    if plotTraining:
                        totalReward = 0

                    # Interact with the training environment until termination
                    while done == 0:

                        # Choose an action according to the RL policy and the current RL state
                        action, _, _ = self.epsilon_greedy_action(state, previousAction)
                        
                        # Interact with the environment with the chosen action
                        nextState, reward, done, info = trainingEnvList[i].step(action)
                        
                        # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                        reward = self.clip_reward(reward)
                        nextState = self.preprocess_state(nextState, coefficients)
                        self.replayMemory.push(state, action, reward, nextState, done)

                        # Trick for better exploration
                        otherAction = int(not bool(action))
                        otherReward = self.clip_reward(info['Reward'])
                        otherNextState = self.preprocess_state(info['State'], coefficients)
                        otherDone = info['Done']
                        self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)

                        # Execute the DQN learning procedure
                        stepsCounter += 1
                        if stepsCounter == learningUpdatePeriod:
                            self.learning()
                            stepsCounter = 0

                        # Update the RL state
                        state = nextState
                        previousAction = action

                        # Continuous tracking of the training performance
                        if plotTraining:
                            totalReward += reward
                    
                    # Store the current training results
                    if plotTraining:
                        score[i][episode] = totalReward
                
                # Compute the current performance on both the training and testing sets
                if plotTraining:
                    # Training set performance
                    trainingEnv = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTrain.append(performance)
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performance, episode)
                    trainingEnv.reset()
                    # Testing set performance
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTest.append(performance)
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performance, episode)
                    testingEnv.reset()
        
        except KeyboardInterrupt:
            print()
            print("WARNING: Training prematurely interrupted...")
            print()
            self.policyNetwork.eval()

        # Assess the algorithm performance on the training trading environment
        trainingEnv = self.testing(trainingEnv, trainingEnv)

        # If required, show the rendering of the trading environment
        if rendering:
            trainingEnv.render()

        # If required, plot the training results
        if plotTraining:
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot(performanceTrain)
            ax.plot(performanceTest)
            ax.legend(["Training", "Testing"])
            plt.savefig(''.join(['Figures/', str(marketSymbol), '_TrainingTestingPerformance', '.png']))
            #plt.show()
            for i in range(len(trainingEnvList)):
                self.plotTraining(score[i][:episode], marketSymbol)

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('TDQN')
        
        # Closing of the tensorboard writer
        self.writer.close()
        
        return trainingEnv


    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False):
        """
        GOAL: Test the RL agent trading policy on a new trading environment
              in order to assess the trading strategy performance.
        
        INPUTS: - trainingEnv: Training RL environment (known).
                - testingEnv: Unknown trading RL environment.
                - rendering: Enable the trading environment rendering.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - testingEnv: Trading environment backtested.
        """

        # Apply data augmentation techniques to process the testing set
        dataAugmentation = DataAugmentation()
        testingEnvSmoothed = dataAugmentation.lowPassFilter(testingEnv, filterOrder)
        trainingEnv = dataAugmentation.lowPassFilter(trainingEnv, filterOrder)

        # Initialization of some RL variables
        coefficients = self.get_features_stats(trainingEnv)
        state = self.preprocess_state(testingEnvSmoothed.reset(), coefficients)
        testingEnv.reset()
        QValues0 = []
        QValues1 = []
        done = 0

        # Interact with the environment until the episode termination
        while done == 0:

            # Choose an action according to the RL policy and the current RL state
            action, _, QValues = self.greedy_action(state)
                
            # Interact with the environment with the chosen action
            nextState, _, done, _ = testingEnvSmoothed.step(action)
            testingEnv.step(action)
                
            # Update the new state
            state = self.preprocess_state(nextState, coefficients)

            # Storing of the Q values
            QValues0.append(QValues[0])
            QValues1.append(QValues[1])

        # If required, show the rendering of the trading environment
        if rendering:
            testingEnv.render()
            self.plotQValues(QValues0, QValues1, testingEnv.marketSymbol)

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance('TDQN')
        
        return testingEnv


    def plotTraining(self, score, marketSymbol):
        """
        GOAL: Plot the training phase results
              (score, sum of rewards).
        
        INPUTS: - score: Array of total episode rewards.
                - marketSymbol: Stock market trading symbol.
        
        OUTPUTS: /
        """

        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Total reward collected', xlabel='Episode')
        ax1.plot(score)
        plt.savefig(''.join(['Figures/', str(marketSymbol), 'TrainingResults', '.png']))
        #plt.show()

    
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


    def plotExpectedPerformance(self, trainingEnv, trainingParameters=[], iterations=10):
        """
        GOAL: Plot the expected performance of the intelligent DRL trading agent.
        
        INPUTS: - trainingEnv: Training RL environment (known).
                - trainingParameters: Additional parameters associated
                                      with the training phase (e.g. the number
                                      of episodes). 
                - iterations: Number of training/testing iterations to compute
                              the expected performance.
        
        OUTPUTS: - trainingEnv: Training RL environment.
        """

        # Preprocessing of the training set
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)

        # Save the initial Deep Neural Network weights
        initialWeights =  copy.deepcopy(self.policyNetwork.state_dict())

        # Initialization of some variables tracking both training and testing performances
        performanceTrain = np.zeros((trainingParameters[0], iterations))
        performanceTest = np.zeros((trainingParameters[0], iterations))

        # Initialization of the testing trading environment
        marketSymbol = trainingEnv.marketSymbol
        startingDate = trainingEnv.endingDate
        endingDate = '2020-1-1'
        money = trainingEnv.data['Money'][0]
        stateLength = trainingEnv.stateLength
        transactionCosts = trainingEnv.transactionCosts
        testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)

        # Print the hardware selected for the training of the Deep Neural Network (either CPU or GPU)
        print("Hardware selected for training: " + str(self.device))
      
        try:

            # Apply the training/testing procedure for the number of iterations specified
            for iteration in range(iterations):

                # Print the progression
                print(''.join(["Expected performance evaluation progression: ", str(iteration+1), "/", str(iterations)]))

                # Training phase for the number of episodes specified as parameter
                for episode in tqdm(range(trainingParameters[0])):

                    # For each episode, train on the entire set of training environments
                    for i in range(len(trainingEnvList)):
                        
                        # Set the initial RL variables
                        coefficients = self.get_features_stats(trainingEnvList[i])
                        trainingEnvList[i].reset()
                        startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                        trainingEnvList[i].setStartingPoint(startingPoint)
                        state = self.preprocess_state(trainingEnvList[i].state, coefficients)
                        previousAction = 0
                        done = 0
                        stepsCounter = 0

                        # Interact with the training environment until termination
                        while done == 0:

                            # Choose an action according to the RL policy and the current RL state
                            action, _, _ = self.epsilon_greedy_action(state, previousAction)
                            
                            # Interact with the environment with the chosen action
                            nextState, reward, done, info = trainingEnvList[i].step(action)

                            # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                            reward = self.clip_reward(reward)
                            nextState = self.preprocess_state(nextState, coefficients)
                            self.replayMemory.push(state, action, reward, nextState, done)

                            # Trick for better exploration
                            otherAction = int(not bool(action))
                            otherReward = self.clip_reward(info['Reward'])
                            otherDone = info['Done']
                            otherNextState = self.preprocess_state(info['State'], coefficients)
                            self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)

                            # Execute the DQN learning procedure
                            stepsCounter += 1
                            if stepsCounter == learningUpdatePeriod:
                                self.learning()
                                stepsCounter = 0

                            # Update the RL state
                            state = nextState
                            previousAction = action
                
                    # Compute both training and testing  current performances
                    trainingEnv = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performanceTrain[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performanceTrain[episode][iteration], episode)     
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performanceTest[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performanceTest[episode][iteration], episode)

                # Restore the initial state of the intelligent RL agent
                if iteration < (iterations-1):
                    trainingEnv.reset()
                    testingEnv.reset()
                    self.policyNetwork.load_state_dict(initialWeights)
                    self.targetNetwork.load_state_dict(initialWeights)
                    self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=learningRate, weight_decay=L2Factor)
                    self.replayMemory.reset()
                    self.iterations = 0
                    stepsCounter = 0
            
            iteration += 1
        
        except KeyboardInterrupt:
            print()
            print("WARNING: Expected performance evaluation prematurely interrupted...")
            print()
            self.policyNetwork.eval()

        # Compute the expected performance of the intelligent DRL trading agent
        expectedPerformanceTrain = []
        expectedPerformanceTest = []
        stdPerformanceTrain = []
        stdPerformanceTest = []
        for episode in range(trainingParameters[0]):
            expectedPerformanceTrain.append(np.mean(performanceTrain[episode][:iteration]))
            expectedPerformanceTest.append(np.mean(performanceTest[episode][:iteration]))
            stdPerformanceTrain.append(np.std(performanceTrain[episode][:iteration]))
            stdPerformanceTest.append(np.std(performanceTest[episode][:iteration]))
        expectedPerformanceTrain = np.array(expectedPerformanceTrain)
        expectedPerformanceTest = np.array(expectedPerformanceTest)
        stdPerformanceTrain = np.array(stdPerformanceTrain)
        stdPerformanceTest = np.array(stdPerformanceTest)

        # Plot each training/testing iteration performance of the intelligent DRL trading agent
        for i in range(iteration):
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot([performanceTrain[e][i] for e in range(trainingParameters[0])])
            ax.plot([performanceTest[e][i] for e in range(trainingParameters[0])])
            ax.legend(["Training", "Testing"])
            plt.savefig(''.join(['Figures/', str(marketSymbol), '_TrainingTestingPerformance', str(i+1), '.png']))
            #plt.show()

        # Plot the expected performance of the intelligent DRL trading agent
        fig = plt.figure()
        ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
        ax.plot(expectedPerformanceTrain)
        ax.plot(expectedPerformanceTest)
        ax.fill_between(range(len(expectedPerformanceTrain)), expectedPerformanceTrain-stdPerformanceTrain, expectedPerformanceTrain+stdPerformanceTrain, alpha=0.25)
        ax.fill_between(range(len(expectedPerformanceTest)), expectedPerformanceTest-stdPerformanceTest, expectedPerformanceTest+stdPerformanceTest, alpha=0.25)
        ax.legend(["Training", "Testing"])
        plt.savefig(''.join(['Figures/', str(marketSymbol), '_TrainingTestingExpectedPerformance', '.png']))
        #plt.show()

        # Closing of the tensorboard writer
        self.writer.close()
        
        return trainingEnv

        
    def saveModel(self, fileName):
        """
        GOAL: Save the RL policy, which is the policy Deep Neural Network.
        
        INPUTS: - fileName: Name of the file.
        
        OUTPUTS: /
        """

        torch.save(self.policyNetwork.state_dict(), fileName)


    def loadModel(self, fileName):
        """
        GOAL: Load a RL policy, which is the policy Deep Neural Network.
        
        INPUTS: - fileName: Name of the file.
        
        OUTPUTS: /
        """

        self.policyNetwork.load_state_dict(torch.load(fileName, map_location=self.device))
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())


    def plotEpsilonAnnealing(self):
        """
        GOAL: Plot the annealing behaviour of the Epsilon variable
              (Epsilon-Greedy exploration technique).
        
        INPUTS: /
        
        OUTPUTS: /
        """

        plt.figure()
        plt.plot([self.epsilonValue(i) for i in range(10*epsilonDecay)])
        plt.xlabel("Iterations")
        plt.ylabel("Epsilon value")
        plt.savefig(''.join(['Figures/', 'EpsilonAnnealing', '.png']))
        #plt.show()
