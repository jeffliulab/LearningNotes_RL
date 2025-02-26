"""
Bandit Algorithms Comparison of
the performance of ε-Greedy, UCB and Gradient Bandit 
to see which method converges fastest and obtains the most rewards.
"""
import numpy as np
import matplotlib.pyplot as plt

# 10-armed bandit environment
class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.q_true = np.random.normal(0, 1, k)  # True reward values for actions
        self.optimal_action = np.argmax(self.q_true)
    
    def get_reward(self, action):
        return np.random.normal(self.q_true[action], 1)  # Reward with Gaussian noise

# Epsilon-Greedy Algorithm
class EpsilonGreedy:
    def __init__(self, bandit, epsilon=0.1, steps=1000):
        self.bandit = bandit
        self.epsilon = epsilon
        self.steps = steps
        self.q_est = np.zeros(bandit.k)
        self.action_count = np.zeros(bandit.k)
    
    def run(self):
        rewards = []
        for t in range(self.steps):
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.bandit.k)  # Exploration
            else:
                action = np.argmax(self.q_est)  # Exploitation
            reward = self.bandit.get_reward(action)
            self.action_count[action] += 1
            self.q_est[action] += (reward - self.q_est[action]) / self.action_count[action]
            rewards.append(reward)
        return np.cumsum(rewards) / np.arange(1, self.steps + 1)  # Average reward over time

# Upper Confidence Bound (UCB) Algorithm
class UCB:
    def __init__(self, bandit, c=2, steps=1000):
        self.bandit = bandit
        self.c = c
        self.steps = steps
        self.q_est = np.zeros(bandit.k)
        self.action_count = np.zeros(bandit.k)
    
    def run(self):
        rewards = []
        for t in range(1, self.steps + 1):
            if 0 in self.action_count:
                action = np.argmin(self.action_count)  # Ensure all actions are tried at least once
            else:
                confidence_bound = self.c * np.sqrt(np.log(t) / self.action_count)
                action = np.argmax(self.q_est + confidence_bound)
            reward = self.bandit.get_reward(action)
            self.action_count[action] += 1
            self.q_est[action] += (reward - self.q_est[action]) / self.action_count[action]
            rewards.append(reward)
        return np.cumsum(rewards) / np.arange(1, self.steps + 1)

# Gradient Bandit Algorithm
class GradientBandit:
    def __init__(self, bandit, alpha=0.1, steps=1000):
        self.bandit = bandit
        self.alpha = alpha
        self.steps = steps
        self.h = np.zeros(bandit.k)  # Preferences
        self.pi = np.ones(bandit.k) / bandit.k  # Initial probabilities
    
    def run(self):
        rewards = []
        avg_reward = 0  # Baseline reward
        for t in range(1, self.steps + 1):
            action = np.random.choice(self.bandit.k, p=self.pi)
            reward = self.bandit.get_reward(action)
            avg_reward += (reward - avg_reward) / t
            self.h -= self.alpha * (reward - avg_reward) * self.pi  # Update all actions
            self.h[action] += self.alpha * (reward - avg_reward) * (1 - self.pi[action])
            self.pi = np.exp(self.h) / np.sum(np.exp(self.h))  # Softmax
            rewards.append(reward)
        return np.cumsum(rewards) / np.arange(1, self.steps + 1)

# Run experiments
steps = 1000
bandit = Bandit()

eps_greedy = EpsilonGreedy(bandit, epsilon=0.1, steps=steps)
ucb = UCB(bandit, c=2, steps=steps)
grad_bandit = GradientBandit(bandit, alpha=0.1, steps=steps)

rewards_eps = eps_greedy.run()
rewards_ucb = ucb.run()
rewards_grad = grad_bandit.run()

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(rewards_eps, label="Epsilon-Greedy (ε=0.1)")
plt.plot(rewards_ucb, label="UCB (c=2)")
plt.plot(rewards_grad, label="Gradient Bandit (α=0.1)")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Comparison of Bandit Algorithms")
plt.legend()
plt.show()
