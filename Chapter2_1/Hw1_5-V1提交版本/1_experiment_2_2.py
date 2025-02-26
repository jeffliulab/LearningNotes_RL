import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar, can be removed if not installed (see comments below)

class Bandit:
    def __init__(self, num_arms=10):
        # The true q*(a) values are sampled from N(0,1)
        self.q_star = np.random.normal(0, 1, num_arms)
    
    def get_reward(self, action):
        # The reward for an action is sampled from N(q*(a),1)
        return np.random.normal(self.q_star[action], 1)
    
    def optimal_action(self):
        # Returns the action with the highest true q*(a) value
        return np.argmax(self.q_star)

class Agent:
    def __init__(self, num_arms=10, epsilon=0.1):
        self.epsilon = epsilon
        self.q_estimates = np.zeros(num_arms)  # Initial Q values set to 0
        self.action_counts = np.zeros(num_arms)  # Track the number of times each action is taken
    
    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.q_estimates))  # Random action (exploration)
        else:
            return np.argmax(self.q_estimates)  # Choose the action with the highest Q estimate (exploitation)
    
    def update(self, action, reward):
        self.action_counts[action] += 1
        alpha = 1 / self.action_counts[action]  # Sample-average update rule
        self.q_estimates[action] += alpha * (reward - self.q_estimates[action])

num_steps = 1000
bandit = Bandit()
agent = Agent(epsilon=0.1)

rewards = []
optimal_action_counts = []

for step in range(num_steps):
    action = agent.select_action()
    reward = bandit.get_reward(action)
    agent.update(action, reward)

    rewards.append(reward)
    optimal_action_counts.append(action == bandit.optimal_action())

print("Final Q estimates:", agent.q_estimates)
print("True q* values:", bandit.q_star)

print("tst bbb")

plt.plot(rewards)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Reward over time")
plt.show(block=False)

print("test aaa")

num_experiments = 2000
epsilons = [0, 0.01, 0.1]
num_arms = 10

print("test running situation")

# Initialize data structures for storing results
avg_rewards = {eps: np.zeros(num_steps) for eps in epsilons}
optimal_action_pct = {eps: np.zeros(num_steps) for eps in epsilons}

# Use tqdm for progress bar, remove it if not needed (replace with "for _ in range(num_experiments):")
# for experiment in range(num_experiments):
# for _ in tqdm(range(num_experiments), desc="Running Experiments"):
for _ in tqdm(range(num_experiments), desc="Running Experiments", ascii=True):
    bandit = Bandit()  # Generate a new bandit problem instance
    
    for eps in epsilons:
        agent = Agent(num_arms=num_arms, epsilon=eps)
        optimal_action = bandit.optimal_action()

        for step in range(num_steps):
            action = agent.select_action()
            reward = bandit.get_reward(action)
            agent.update(action, reward)

            avg_rewards[eps][step] += reward
            optimal_action_pct[eps][step] += (action == optimal_action)

# Compute the final average over all experiments
for eps in epsilons:
    avg_rewards[eps] /= num_experiments
    optimal_action_pct[eps] = (optimal_action_pct[eps] / num_experiments) * 100

print("Finished 2000 experiments!")

# Plot results
plt.figure(figsize=(12, 5))

# Plot average reward over steps
plt.subplot(1, 2, 1)
for eps in epsilons:
    plt.plot(avg_rewards[eps], label=f'ε={eps}')
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()
plt.title("Average Reward vs Steps")

# Plot optimal action selection percentage over steps
plt.subplot(1, 2, 2)
for eps in epsilons:
    plt.plot(optimal_action_pct[eps], label=f'ε={eps}')
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.legend()
plt.title("Optimal Action Selection vs Steps")

plt.show()
