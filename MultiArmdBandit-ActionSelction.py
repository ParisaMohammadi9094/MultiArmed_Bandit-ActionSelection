import numpy as np
import matplotlib.pyplot as plt
import random

class MultiArmedBandit:
    def __init__(self, k):
        self.k = k
        self.probs = np.random.rand(k)

    def pull(self, arm):
        return 1 if np.random.rand() < self.probs[arm] else 0

class Agent:
    def __init__(self, k):
        self.k = k
        self.counts = np.zeros(k)
        self.values = np.zeros(k)

    def select_action(self):
        pass

    def update(self, action, reward):
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]

class GreedyAgent(Agent):
    def select_action(self):
        return np.argmax(self.values)

class EpsilonGreedyAgent(Agent):
    def __init__(self, k, epsilon):
        super().__init__(k)
        self.epsilon = epsilon

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.values)

class UCBAgent(Agent):
    def __init__(self, k, c):
        super().__init__(k)
        self.c = c
        self.t = 0

    def select_action(self):
        self.t += 1
        if 0 in self.counts:
            return np.argmin(self.counts)
        ucb_values = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return np.argmax(ucb_values)

class RandomAgent(Agent):
    def select_action(self):
        return np.random.randint(self.k)

class BoltzmannAgent(Agent):
    def __init__(self, k, temperature):
        super().__init__(k)
        self.temperature = temperature

    def select_action(self):
        exp_values = np.exp(self.values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(np.arange(self.k), p=probabilities)

class BayesianAgent(Agent):
    def __init__(self, k):
        super().__init__(k)
        self.alpha = np.ones(k)
        self.beta = np.ones(k)

    def select_action(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, action, reward):
        super().update(action, reward)
        self.alpha[action] += reward
        self.beta[action] += 1 - reward

def simulate(agent, bandit, episodes):
    rewards = np.zeros(episodes)
    for i in range(episodes):
        action = agent.select_action()
        reward = bandit.pull(action)
        agent.update(action, reward)
        rewards[i] = reward
    return rewards

def plot_results(results, labels):
    for result, label in zip(results, labels):
        plt.plot(np.cumsum(result), label=label)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()

k = 10
episodes = 200000
bandit = MultiArmedBandit(k)

agents = [
    GreedyAgent(k),
    EpsilonGreedyAgent(k, epsilon=0.1),
    UCBAgent(k, c=2),
    RandomAgent(k),
    BoltzmannAgent(k, temperature=0.1),
    BayesianAgent(k)
]

labels = ['Greedy', 'Epsilon-Greedy', 'UCB', 'Random', 'Boltzmann', 'Bayesian']

results = [simulate(agent, bandit, episodes) for agent in agents]

plot_results(results, labels)
