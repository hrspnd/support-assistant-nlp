# rl_agent.py
# Multi-Armed Bandit (epsilon-greedy)

import random


class MultiArmedBandit:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.counts = [0] * n_actions
        self.values = [0.0] * n_actions

    def select_action(self):
        # exploration vs exploitation
        epsilon = 0.5

        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)

        return self.values.index(max(self.values))

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]

        # incremental update
        self.values[action] = value + (reward - value) / n