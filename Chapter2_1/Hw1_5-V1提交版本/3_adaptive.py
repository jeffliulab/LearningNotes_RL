import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar for visualization

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

# Agent (ε-greedy strategy, supports three Q estimation methods)
class Agent:
    def __init__(self, num_arms=10, epsilon=0.1, alpha=None, adaptive=False):
        self.epsilon = epsilon
        self.alpha = alpha  # None for sample-average method, fixed value for constant step-size
        self.adaptive = adaptive  # If True, uses adaptive step-size α_t = 1/t
        self.q_estimates = np.zeros(num_arms)  # Estimated Q values
        self.action_counts = np.zeros(num_arms)  # Track count of each action taken

    def select_action(self):
        """ ε-greedy action selection """
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.q_estimates))  # 10% exploration (random action)
        else:
            return np.argmax(self.q_estimates)  # 90% exploitation (choose best-known action)

    def update(self, action, reward):
        """ Update Q estimates """
        self.action_counts[action] += 1
        if self.adaptive:  # Adaptive step-size α_t = 1/t
            alpha = 1 / self.action_counts[action]
        elif self.alpha is None:  # Sample-average method α_t = 1/N_t
            alpha = 1 / self.action_counts[action]
        else:  # Fixed step-size α = 0.1
            alpha = self.alpha
        self.q_estimates[action] += alpha * (reward - self.q_estimates[action])

# Run experiment
def run_experiment(num_steps=10000, num_experiments=2000, epsilon=0.1, alpha=None, adaptive=False):
    avg_rewards = np.zeros(num_steps)
    optimal_action_pct = np.zeros(num_steps)

    for _ in tqdm(range(num_experiments), desc="Running Experiments"):
        bandit = Bandit()
        agent = Agent(epsilon=epsilon, alpha=alpha, adaptive=adaptive)
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

# Run experiment and compare three Q estimation methods
num_steps = 10000
num_experiments = 2000
epsilon = 0.1

# 1️⃣ Sample-average method
print("Sample Average - Incrementally Computed: ")
rewards_sample_avg, optimal_sample_avg = run_experiment(num_steps, num_experiments, epsilon, alpha=None)

# 2️⃣ Fixed step-size α=0.1
print("Fixed Step-Size α=0.1: ")
rewards_const_alpha, optimal_const_alpha = run_experiment(num_steps, num_experiments, epsilon, alpha=0.1)

# 3️⃣ Adaptive step-size α_t = 1/t
print("Adaptive Step-Size α_t = 1/t: ")
rewards_adaptive, optimal_adaptive = run_experiment(num_steps, num_experiments, epsilon, adaptive=True)

# 🎨 **Plot results**
plt.figure(figsize=(12, 5))

# Average reward curve
plt.subplot(1, 2, 1)
plt.plot(rewards_sample_avg, linestyle="-", linewidth=1.5, label='Sample-Average')
plt.plot(rewards_const_alpha, linestyle="--", linewidth=1.5, label='Fixed Step-Size α=0.1')
plt.plot(rewards_adaptive, linestyle=":", linewidth=1.5, label='Adaptive Step-Size α_t=1/t')
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()
plt.title("Average Reward vs Steps")

# Optimal action selection rate curve
plt.subplot(1, 2, 2)
plt.plot(optimal_sample_avg, linestyle="-", linewidth=1.5, label='Sample-Average')
plt.plot(optimal_const_alpha, linestyle="--", linewidth=1.5, label='Fixed Step-Size α=0.1')
plt.plot(optimal_adaptive, linestyle=":", linewidth=1.5, label='Adaptive Step-Size α_t=1/t')
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

# Smoothed average reward curve
plt.subplot(1, 2, 1)
plt.plot(x_smooth, moving_average(rewards_sample_avg), linestyle="-", linewidth=2.5, label='Sample-Average')
plt.plot(x_smooth, moving_average(rewards_const_alpha), linestyle="--", linewidth=2.5, label='Fixed Step-Size α=0.1')
plt.plot(x_smooth, moving_average(rewards_adaptive), linestyle=":", linewidth=2.5, label='Adaptive Step-Size α_t=1/t')
plt.xlabel("Steps")
plt.ylabel("Smoothed Average Reward")
plt.legend()
plt.title("Smoothed Average Reward vs Steps")

# Smoothed optimal action selection rate curve
plt.subplot(1, 2, 2)
plt.plot(x_smooth, moving_average(optimal_sample_avg), linestyle="-", linewidth=2.5, label='Sample-Average')
plt.plot(x_smooth, moving_average(optimal_const_alpha), linestyle="--", linewidth=2.5, label='Fixed Step-Size α=0.1')
plt.plot(x_smooth, moving_average(optimal_adaptive), linestyle=":", linewidth=2.5, label='Adaptive Step-Size α_t=1/t')
plt.xlabel("Steps")
plt.ylabel("Smoothed % Optimal Action")
plt.legend()
plt.title("Smoothed Optimal Action Selection vs Steps")

plt.show()
