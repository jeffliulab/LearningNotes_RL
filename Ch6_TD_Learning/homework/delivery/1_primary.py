import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class GridWorldEnv:
    def __init__(self):
        self.rows = 10
        self.cols = 10
        
        # Start and goal positions (row, col)
        self.start_pos = (5, 3)
        self.end_pos = (5, 8)
        
        # Base wind strength for each column
        self.base_wind = np.array([1, 1, 1, 2, 2, 2, 2, 1, 1, 1])
        
        # Actions: 8 directions + stay
        self.actions = [(0, 0), (-1, 0), (-1, 1), (0, 1), (1, 1), 
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.n_actions = len(self.actions)
        
        self.position = self.start_pos
        self.steps = 0
        
    def reset(self):
        self.position = self.start_pos
        self.steps = 0
        return self.position
    
    def get_stochastic_wind(self, col):
        wind = self.base_wind[col]
        rand = np.random.random()
        if rand < 1/3:
            wind += 1
        elif rand < 2/3:
            wind = wind  # unchanged
        else:
            wind = max(0, wind - 1)
        return wind
    
    def step(self, action):
        self.steps += 1
        row_delta, col_delta = self.actions[action]
        
        row, col = self.position
        new_row = row + row_delta
        new_col = col + col_delta
        
        # Wind pushes upward (reduces row)
        wind = self.get_stochastic_wind(col)
        new_row = new_row - wind
        
        # Keep within grid boundaries
        new_row = max(0, min(new_row, self.rows - 1))
        new_col = max(0, min(new_col, self.cols - 1))
        
        self.position = (new_row, new_col)
        done = self.position == self.end_pos
        reward = 0 if done else -1
        
        return self.position, reward, done
        
class Agent:
    def __init__(self, env, algorithm='sarsa', epsilon=0.1, alpha=0.5, gamma=0.95):
        self.env = env
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        # Initialize Q-values table
        self.Q = np.zeros((env.rows, env.cols, env.n_actions))
        
        # For Monte Carlo learning
        self.returns = {}
        self.episode = []
    
    def choose_action(self, state, exploring=True):
        row, col = state
        if exploring and np.random.random() < self.epsilon:
            return np.random.randint(self.env.n_actions)
        else:
            return np.argmax(self.Q[row, col])
    
    def sarsa_update(self, state, action, reward, next_state, next_action):
        row, col = state
        next_row, next_col = next_state
        self.Q[row, col, action] += self.alpha * (
            reward + self.gamma * self.Q[next_row, next_col, next_action] - self.Q[row, col, action]
        )
    
    def q_learning_update(self, state, action, reward, next_state):
        row, col = state
        next_row, next_col = next_state
        max_next_q = np.max(self.Q[next_row, next_col])
        self.Q[row, col, action] += self.alpha * (
            reward + self.gamma * max_next_q - self.Q[row, col, action]
        )
    
    def mc_store(self, state, action, reward):
        self.episode.append(((state[0], state[1]), action, reward))
    
    def mc_update(self):
        G = 0
        visited = set()
        for state, action, reward in reversed(self.episode):
            G = self.gamma * G + reward
            state_action = (state, action)
            if state_action not in visited:
                visited.add(state_action)
                if state_action not in self.returns:
                    self.returns[state_action] = []
                self.returns[state_action].append(G)
                row, col = state
                self.Q[row, col, action] = np.mean(self.returns[state_action])
        self.episode = []

def run_experiment(n_episodes=1000, n_runs=10):
    sarsa_rewards = np.zeros((n_runs, n_episodes))
    qlearning_rewards = np.zeros((n_runs, n_episodes))
    mc_rewards = np.zeros((n_runs, n_episodes))
    
    sarsa_success = np.zeros((n_runs, n_episodes))
    qlearning_success = np.zeros((n_runs, n_episodes))
    mc_success = np.zeros((n_runs, n_episodes))
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        env = GridWorldEnv()
        sarsa_agent = Agent(env, algorithm='sarsa')
        qlearning_agent = Agent(env, algorithm='q_learning')
        mc_agent = Agent(env, algorithm='mc')
        
        # Train SARSA
        for episode in tqdm(range(n_episodes), desc="SARSA"):
            state = env.reset()
            action = sarsa_agent.choose_action(state)
            total_reward = 0
            done = False
            max_steps = 1000
            
            while not done and env.steps < max_steps:
                next_state, reward, done = env.step(action)
                next_action = sarsa_agent.choose_action(next_state)
                sarsa_agent.sarsa_update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
                total_reward += reward
            
            sarsa_rewards[run, episode] = total_reward
            sarsa_success[run, episode] = 1 if done else 0
        
        # Train Q-learning
        for episode in tqdm(range(n_episodes), desc="Q-learning"):
            state = env.reset()
            total_reward = 0
            done = False
            max_steps = 1000
            
            while not done and env.steps < max_steps:
                action = qlearning_agent.choose_action(state)
                next_state, reward, done = env.step(action)
                qlearning_agent.q_learning_update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            
            qlearning_rewards[run, episode] = total_reward
            qlearning_success[run, episode] = 1 if done else 0
        
        # Train Monte Carlo
        for episode in tqdm(range(n_episodes), desc="Monte Carlo"):
            state = env.reset()
            total_reward = 0
            done = False
            max_steps = 1000
            
            while not done and env.steps < max_steps:
                action = mc_agent.choose_action(state)
                next_state, reward, done = env.step(action)
                mc_agent.mc_store(state, action, reward)
                state = next_state
                total_reward += reward
            
            if env.steps >= max_steps and not done:
                mc_agent.episode.append(((state[0], state[1]), 0, -10))
            
            mc_agent.mc_update()
            mc_rewards[run, episode] = total_reward
            mc_success[run, episode] = 1 if done else 0
    
    sarsa_avg_rewards = np.mean(sarsa_rewards, axis=0)
    qlearning_avg_rewards = np.mean(qlearning_rewards, axis=0)
    mc_avg_rewards = np.mean(mc_rewards, axis=0)
    
    sarsa_avg_success = np.mean(sarsa_success, axis=0)
    qlearning_avg_success = np.mean(qlearning_success, axis=0)
    mc_avg_success = np.mean(mc_success, axis=0)
    
    window_size = 10
    def smooth(data, window_size):
        smoothed = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window_size)
            smoothed[i] = np.mean(data[start:i+1])
        return smoothed
    
    sarsa_smooth_rewards = smooth(sarsa_avg_rewards, window_size)
    qlearning_smooth_rewards = smooth(qlearning_avg_rewards, window_size)
    mc_smooth_rewards = smooth(mc_avg_rewards, window_size)
    
    sarsa_smooth_success = smooth(sarsa_avg_success, window_size)
    qlearning_smooth_success = smooth(qlearning_avg_success, window_size)
    mc_smooth_success = smooth(mc_avg_success, window_size)
    
    # Plot rewards vs episodes with different line styles and markers
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sarsa_smooth_rewards, linestyle='-', marker='o', label='SARSA')
    plt.plot(qlearning_smooth_rewards, linestyle='--', marker='s', label='Q-learning')
    plt.plot(mc_smooth_rewards, linestyle='-.', marker='^', label='Monte Carlo')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Reward vs Episodes')
    plt.legend()
    plt.grid(True)
    
    # Plot success rate vs episodes with different line styles and markers
    plt.subplot(1, 2, 2)
    plt.plot(sarsa_smooth_success, linestyle='-', marker='o', label='SARSA')
    plt.plot(qlearning_smooth_success, linestyle='--', marker='s', label='Q-learning')
    plt.plot(mc_smooth_success, linestyle='-.', marker='^', label='Monte Carlo')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs Episodes')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()

    print("\nFinal Statistics (after smoothing):")
    print(f"SARSA - Final Reward: {sarsa_smooth_rewards[-1]:.2f}, Success Rate: {sarsa_smooth_success[-1]:.2f}")
    print(f"Q-learning - Final Reward: {qlearning_smooth_rewards[-1]:.2f}, Success Rate: {qlearning_smooth_success[-1]:.2f}")
    print(f"Monte Carlo - Final Reward: {mc_smooth_rewards[-1]:.2f}, Success Rate: {mc_smooth_success[-1]:.2f}")

if __name__ == "__main__":
    run_experiment(n_episodes=500, n_runs=3)
