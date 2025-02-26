"""
Simulation of the recommendation system, 
using the gradient bandit to adjust the recommendation probability to increase the user click rate.
"""

import numpy as np
import matplotlib.pyplot as plt

# Recommendation System using Gradient Bandit
class RecommendationSystem:
    def __init__(self, num_items=10, alpha=0.1, steps=1000):
        self.num_items = num_items
        self.alpha = alpha
        self.steps = steps
        self.h = np.zeros(num_items)  # Preferences for items
        self.pi = np.ones(num_items) / num_items  # Initial probabilities
        self.true_rewards = np.random.normal(0, 1, num_items)  # Simulated user preferences
    
    def get_reward(self, item):
        return np.random.normal(self.true_rewards[item], 1)  # Simulated user feedback
    
    def run(self):
        rewards = []
        avg_reward = 0
        for t in range(1, self.steps + 1):
            item = np.random.choice(self.num_items, p=self.pi)
            reward = self.get_reward(item)
            avg_reward += (reward - avg_reward) / t
            self.h -= self.alpha * (reward - avg_reward) * self.pi  # Update all items
            self.h[item] += self.alpha * (reward - avg_reward) * (1 - self.pi[item])
            self.pi = np.exp(self.h) / np.sum(np.exp(self.h))  # Softmax
            rewards.append(reward)
        return np.cumsum(rewards) / np.arange(1, self.steps + 1)

# Run recommendation system simulation
recommender = RecommendationSystem()
rewards_recommender = recommender.run()

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(rewards_recommender, label="Recommendation System (Gradient Bandit)")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Recommendation System using Gradient Bandit Algorithm")
plt.legend()
plt.show()
