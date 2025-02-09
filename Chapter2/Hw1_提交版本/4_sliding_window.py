import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar for visualization
from collections import deque  # Used for sliding window implementation

# 10-armed bandit environment (non-stationary setting)
class Bandit:
    def __init__(self, num_arms=10):
        self.num_arms = num_arms
        self.q_star = np.zeros(num_arms)  # Initialize all q*(a) to 0

    def get_reward(self, action):
        return np.random.normal(self.q_star[action], 1)  # Reward sampled from N(q*(a), 1)

    def update_q_star(self):
        """ Perform random walk for all q*(a) at each step """
        self.q_star += np.random.normal(0, 0.01, self.num_arms)

    def optimal_action(self):
        """ Return the index of the arm with the highest q*(a) """
        return np.argmax(self.q_star)


# Agent class with Sliding Window method
class Agent:
    def __init__(self, num_arms=10, epsilon=0.1, alpha=None, window_size=None):
        self.epsilon = epsilon
        self.alpha = alpha
        self.window_size = window_size
        self.q_estimates = np.zeros(num_arms)  # Estimated Q values
        self.action_counts = np.zeros(num_arms)  # Track count of each action taken
        self.reward_windows = [deque(maxlen=window_size) for _ in range(num_arms)] if window_size else None

    def select_action(self):
        """ ε-greedy action selection """
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.q_estimates))  # 10% exploration (random action)
        else:
            return np.argmax(self.q_estimates)  # 90% exploitation (choose best-known action)

    def update(self, action, reward):
        """ Update Q estimates """
        self.action_counts[action] += 1

        if self.window_size:  # Sliding window update
            self.reward_windows[action].append(reward)
            self.q_estimates[action] = np.mean(self.reward_windows[action])
        elif self.alpha == "1/t":  # Adaptive step-size α_t = 1/t
            alpha = 1 / self.action_counts[action]
            self.q_estimates[action] += alpha * (reward - self.q_estimates[action])
        elif self.alpha is None:  # Sample-average method
            alpha = 1 / self.action_counts[action]
            self.q_estimates[action] += alpha * (reward - self.q_estimates[action])
        else:  # Fixed step-size α=0.1
            self.q_estimates[action] += self.alpha * (reward - self.q_estimates[action])


# Run experiment
def run_experiment(num_steps=10000, num_experiments=2000, epsilon=0.1, alpha=None, window_size=None):
    avg_rewards = np.zeros(num_steps)
    optimal_action_pct = np.zeros(num_steps)

    for _ in tqdm(range(num_experiments), desc="Running Experiments"):
        bandit = Bandit()
        agent = Agent(epsilon=epsilon, alpha=alpha, window_size=window_size)
        optimal_action = bandit.optimal_action()

        for step in range(num_steps):
            action = agent.select_action()
            reward = bandit.get_reward(action)
            agent.update(action, reward)
            bandit.update_q_star()  # Apply random walk to q*(a)

            avg_rewards[step] += reward
            optimal_action_pct[step] += (action == bandit.optimal_action())

    avg_rewards /= num_experiments
    optimal_action_pct = (optimal_action_pct / num_experiments) * 100  # Convert to percentage
    return avg_rewards, optimal_action_pct


# ** Run experiment**
num_steps = 10000
num_experiments = 2000
epsilon = 0.1

# Sample-average method
print("Sample Average - Incrementally Computed: ")
rewards_sample_avg, optimal_sample_avg = run_experiment(num_steps, num_experiments, epsilon, alpha=None)

# Fixed step-size α=0.1
print("Fixed Step-Size α=0.1: ")
rewards_const_alpha, optimal_const_alpha = run_experiment(num_steps, num_experiments, epsilon, alpha=0.1)

# Adaptive step-size α_t = 1/t
print("Adaptive Step-Size α_t = 1/t: ")
rewards_adaptive, optimal_adaptive = run_experiment(num_steps, num_experiments, epsilon, alpha="1/t")

# Sliding window (N=100)
print("Sliding Window (N=100):")
rewards_window, optimal_window = run_experiment(num_steps, num_experiments, epsilon, window_size=100)


# **Plot results**
plt.figure(figsize=(12, 5))

# **Average reward curve**
plt.subplot(1, 2, 1)
plt.plot(rewards_sample_avg, linestyle="-", linewidth=1.5, label='Sample-Average')
plt.plot(rewards_const_alpha, linestyle="--", linewidth=1.5, label='Fixed Step-Size α=0.1')
plt.plot(rewards_adaptive, linestyle=":", linewidth=1.5, label='Adaptive Step-Size α_t=1/t')
plt.plot(rewards_window, linestyle="-.", linewidth=1.5, label='Sliding Window (N=100)')
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()
plt.title("Average Reward vs Steps")

# **Optimal action selection rate curve**
plt.subplot(1, 2, 2)
plt.plot(optimal_sample_avg, linestyle="-", linewidth=1.5, label='Sample-Average')
plt.plot(optimal_const_alpha, linestyle="--", linewidth=1.5, label='Fixed Step-Size α=0.1')
plt.plot(optimal_adaptive, linestyle=":", linewidth=1.5, label='Adaptive Step-Size α_t=1/t')
plt.plot(optimal_window, linestyle="-.", linewidth=1.5, label='Sliding Window (N=100)')
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.legend()
plt.title("Optimal Action Selection vs Steps")

plt.show()



# **New Plots: Using Moving Average for a Smoother Trend**
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

x_smooth = np.arange(len(moving_average(rewards_sample_avg)))

plt.figure(figsize=(12, 5))

markers = ["o", "s", "^", "x"] 
labels = [
    "Sample Average",
    "Fixed Step-Size α=0.1",
    "Adaptive Step-Size α_t=1/t",
    "Sliding Window (N=100)"
]
colors = ["blue", "orange", "green", "red"] 
data = [
    (rewards_sample_avg, optimal_sample_avg),
    (rewards_const_alpha, optimal_const_alpha),
    (rewards_adaptive, optimal_adaptive),
    (rewards_window, optimal_window)
]

# Smoothed average reward curve
plt.subplot(1, 2, 1)
for (rewards, _), marker, label, color in zip(data, markers, labels, colors):
    plt.plot(x_smooth, moving_average(rewards), marker=marker, markevery=500, linewidth=2.5, label=label, color=color)
plt.xlabel("Steps")
plt.ylabel("Smoothed Average Reward")
plt.legend()
plt.title("Smoothed Average Reward vs Steps")

# Smoothed optimal action selection rate curve
plt.subplot(1, 2, 2)
for (_, optimal), marker, label, color in zip(data, markers, labels, colors):
    plt.plot(x_smooth, moving_average(optimal), marker=marker, markevery=500, linewidth=2.5, label=label, color=color)
plt.xlabel("Steps")
plt.ylabel("Smoothed % Optimal Action")
plt.legend()
plt.title("Smoothed Optimal Action Selection vs Steps")

plt.show()


