# rl_agent.py
# Multi-armed bandit stub with simulated reward probabilities

import random
import matplotlib.pyplot as plt


class MultiArmedBandit:

    def __init__(self, responses):
        self.responses = responses
        self.counts = [0] * len(responses)
        self.values = [0.0] * len(responses)

    def select_action(self):
        # choose the response with highest estimated value
        if random.random() < 0.2:  # exploration
            return random.randint(0, len(self.responses) - 1)
        return self.values.index(max(self.values))

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]

        new_value = value + (reward - value) / n
        self.values[action] = new_value


responses = [
    "Please provide your order number.",
    "Let me check the status of your delivery.",
    "I can help track your package."
]

# simulate environment quality of responses
true_reward_probabilities = [0.7, 0.5, 0.3] # PLACEHOLDER FOR REINFORCED LEARNING. not pure random, but not real results.

agent = MultiArmedBandit(responses)

avg_rewards = []
total_reward = 0
episodes = 100

for episode in range(episodes):

    action = agent.select_action()

    prob = true_reward_probabilities[action]
    reward = 1 if random.random() < prob else 0

    agent.update(action, reward)

    total_reward += reward
    avg_rewards.append(total_reward / (episode + 1))


plt.plot(avg_rewards)
plt.title("RL Agent Average Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid(True)
plt.show()
