"""
最终解决的问题：

gridworld

网格世界大小：10 x 10 十行十列

起始点：start point （3，5） 第三列
终点：end point （8，5） 第八列

风向：每一列都有一个固定的风力强度
固定强度：头三列为1，4-7列为2，8-10列为1
在固定强度的基础上，还有一定几率发生变化：
1、1/3的几率 +1
2、1/3的几率 保持不变
3、1/3的几率 -1

action：可以有八个方向的移动 + 保持原地不动 共计9种选项

epsilon = 0.1

alpha = 0.5

gamma = 0.95


对比MC，Sarsa和Q-learning在该问题求解上的学习曲线

学习曲线包括两种：
1、 (reward vs time) 
2、(success rate vs time

记得在图像上增加其他标识

primary question: 
两个excersice综合的一道大题


"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class GridWorldEnv:
    def __init__(self):
        # Grid size
        self.rows = 10
        self.cols = 10
        
        # Start and end positions
        self.start_pos = (5, 3)  # (row, col) format
        self.end_pos = (5, 8)
        
        # Wind strengths for each column
        self.base_wind = np.array([1, 1, 1, 2, 2, 2, 2, 1, 1, 1])
        
        # Actions: 8 directions + stay
        # 0: stay, 1: up, 2: up-right, 3: right, 4: down-right, 
        # 5: down, 6: down-left, 7: left, 8: up-left
        self.actions = [(0, 0), (-1, 0), (-1, 1), (0, 1), (1, 1), 
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.n_actions = len(self.actions)
        
        # Current position
        self.position = self.start_pos
        self.steps = 0
        
    def reset(self):
        # Reset position to start
        self.position = self.start_pos
        self.steps = 0
        return self.position
    
    def get_stochastic_wind(self, col):
        # Get base wind strength for the column
        wind = self.base_wind[col]
        
        # Apply stochasticity
        rand = np.random.random()
        if rand < 1/3:
            wind += 1
        elif rand < 2/3:
            wind = wind  # Unchanged
        else:
            wind = max(0, wind - 1)  # Prevent negative wind
            
        return wind
    
    def step(self, action):
        # Increment step counter
        self.steps += 1
        
        # Get action movement
        row_delta, col_delta = self.actions[action]
        
        # Apply action
        row, col = self.position
        new_row = row + row_delta
        new_col = col + col_delta
        
        # Apply wind (wind pushes upward, reducing row)
        wind = self.get_stochastic_wind(col)
        new_row = new_row - wind
        
        # Bound within grid
        new_row = max(0, min(new_row, self.rows - 1))
        new_col = max(0, min(new_col, self.cols - 1))
        
        # Update position
        self.position = (new_row, new_col)
        
        # Check if reached the goal
        done = self.position == self.end_pos
        
        # Reward: -1 per step, 0 when reaching the goal
        reward = 0 if done else -1
        
        return self.position, reward, done
        
class Agent:
    def __init__(self, env, algorithm='sarsa', epsilon=0.1, alpha=0.5, gamma=0.95):
        self.env = env
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        # Initialize Q-values
        self.Q = np.zeros((env.rows, env.cols, env.n_actions))
        
        # For MC
        self.returns = {}  # state-action -> returns
        self.episode = []  # store state, action, reward
    
    def choose_action(self, state, exploring=True):
        row, col = state
        
        # Epsilon-greedy policy
        if exploring and np.random.random() < self.epsilon:
            return np.random.randint(self.env.n_actions)
        else:
            return np.argmax(self.Q[row, col])
    
    def sarsa_update(self, state, action, reward, next_state, next_action):
        row, col = state
        next_row, next_col = next_state
        
        # SARSA update
        self.Q[row, col, action] += self.alpha * (
            reward + self.gamma * self.Q[next_row, next_col, next_action] - self.Q[row, col, action]
        )
    
    def q_learning_update(self, state, action, reward, next_state):
        row, col = state
        next_row, next_col = next_state
        
        # Q-learning update
        max_next_q = np.max(self.Q[next_row, next_col])
        self.Q[row, col, action] += self.alpha * (
            reward + self.gamma * max_next_q - self.Q[row, col, action]
        )
    
    def mc_store(self, state, action, reward):
        # Store state as tuple to ensure it's hashable
        self.episode.append(((state[0], state[1]), action, reward))
    
    def mc_update(self):
        # Process episode
        G = 0
        visited = set()
        
        # Going backwards through the episode
        for state, action, reward in reversed(self.episode):
            G = self.gamma * G + reward
            
            # Ensure first-visit MC - state is already a hashable tuple
            state_action = (state, action)
            if state_action not in visited:
                visited.add(state_action)
                
                # Update returns
                if state_action not in self.returns:
                    self.returns[state_action] = []
                self.returns[state_action].append(G)
                
                # Update Q-value with the average return
                row, col = state
                self.Q[row, col, action] = np.mean(self.returns[state_action])
        
        # Clear episode
        self.episode = []

def run_experiment(n_episodes=1000, n_runs=10):
    # Store results from all runs
    sarsa_rewards = np.zeros((n_runs, n_episodes))
    qlearning_rewards = np.zeros((n_runs, n_episodes))
    mc_rewards = np.zeros((n_runs, n_episodes))
    
    sarsa_success = np.zeros((n_runs, n_episodes))
    qlearning_success = np.zeros((n_runs, n_episodes))
    mc_success = np.zeros((n_runs, n_episodes))
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        
        # Create environment and agents
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
            
            # Set max steps to prevent infinite loops
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
            
            # Set max steps to prevent infinite loops
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
            
            # Set max steps to prevent infinite loops
            max_steps = 1000
            
            # Generate episode
            while not done and env.steps < max_steps:
                action = mc_agent.choose_action(state)
                next_state, reward, done = env.step(action)
                
                mc_agent.mc_store(state, action, reward)
                
                state = next_state
                total_reward += reward
            
            # Handle case where max steps reached
            if env.steps >= max_steps and not done:
                # Add a termination signal for MC
                # This is artificial but helps MC learn when episodes don't naturally terminate
                mc_agent.episode.append(((state[0], state[1]), 0, -10))  # Penalty for timeout
            
            # Update Q-values after episode completion
            mc_agent.mc_update()
            
            mc_rewards[run, episode] = total_reward
            mc_success[run, episode] = 1 if done else 0
    
    # Average results across runs
    sarsa_avg_rewards = np.mean(sarsa_rewards, axis=0)
    qlearning_avg_rewards = np.mean(qlearning_rewards, axis=0)
    mc_avg_rewards = np.mean(mc_rewards, axis=0)
    
    sarsa_avg_success = np.mean(sarsa_success, axis=0)
    qlearning_avg_success = np.mean(qlearning_success, axis=0)
    mc_avg_success = np.mean(mc_success, axis=0)
    
    # Smoothing for better visualization
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
    
    # Plot rewards vs time
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sarsa_smooth_rewards, label='SARSA')
    plt.plot(qlearning_smooth_rewards, label='Q-learning')
    plt.plot(mc_smooth_rewards, label='Monte Carlo')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Reward vs Episodes')
    plt.legend()
    plt.grid(True)
    
    # Plot success rate vs time
    plt.subplot(1, 2, 2)
    plt.plot(sarsa_smooth_success, label='SARSA')
    plt.plot(qlearning_smooth_success, label='Q-learning')
    plt.plot(mc_smooth_success, label='Monte Carlo')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs Episodes')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()

    # Print final statistics
    print("\nFinal Statistics (after smoothing):")
    print(f"SARSA - Final Reward: {sarsa_smooth_rewards[-1]:.2f}, Success Rate: {sarsa_smooth_success[-1]:.2f}")
    print(f"Q-learning - Final Reward: {qlearning_smooth_rewards[-1]:.2f}, Success Rate: {qlearning_smooth_success[-1]:.2f}")
    print(f"Monte Carlo - Final Reward: {mc_smooth_rewards[-1]:.2f}, Success Rate: {mc_smooth_success[-1]:.2f}")

# Run experiment
if __name__ == "__main__":
    # You can adjust parameters as needed
    run_experiment(n_episodes=500, n_runs=3)