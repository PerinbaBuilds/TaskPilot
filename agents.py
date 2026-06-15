import numpy as np


class Agent:
    def __init__(self, name, lr=0.05):
        self.name = name
        self.lr = lr
        self.weights = np.random.randn(4)

    def act(self, state):
        return float(np.dot(self.weights, state))

    def learn(self, state, reward):
        pred = self.act(state)
        error = reward - pred
        self.weights += self.lr * error * state
