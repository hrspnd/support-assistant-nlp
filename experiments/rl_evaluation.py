# experiments/rl_evaluation.py
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rl_agent import MultiArmedBandit



def simulate_bandit(steps=500):
    """
    Simulates interaction with the bandit using a fake environment.
    Returns rewards and bandit instance.
    """

    # True reward probabilities for each action (hidden from the agent)
    # You can adjust these to simulate different scenarios
    true_rewards = [0.2, 0.5, 0.8, 0.3]

    bandit = MultiArmedBandit(n_actions=len(true_rewards))

    rewards = []

    for step in range(steps):
        action = bandit.select_action()

        # Simulated reward (1 or 0 based on probability)
        reward = 1 if np.random.rand() < true_rewards[action] else 0

        bandit.update(action, reward)
        rewards.append(reward)

    return rewards, bandit


def plot_learning_curve(rewards):
    """Plots and saves the learning curve (average reward over time)."""

    avg_rewards = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

    plt.figure()
    plt.plot(avg_rewards)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("RL Bandit Learning Curve")
    plt.grid()

    # Save figure for your paper
    plt.savefig("experiments/rl_learning_curve.png")
    plt.show()


def plot_action_distribution(bandit):
    """Plots how often each action was selected."""

    plt.figure()
    plt.bar(range(len(bandit.counts)), bandit.counts)
    plt.xlabel("Actions")
    plt.ylabel("Selection Count")
    plt.title("Action Selection Distribution")
    plt.grid()

    # Save figure
    plt.savefig("experiments/rl_action_distribution.png")
    plt.show()


def main():
    print("Running RL Bandit Simulation...")

    rewards, bandit = simulate_bandit(steps=500)

    print("Plotting learning curve...")
    plot_learning_curve(rewards)

    print("Plotting action distribution...")
    plot_action_distribution(bandit)

    print("Done. Graphs saved in /experiments folder.")


if __name__ == "__main__":
    main()