"""
This code implements a discrete version of the Breakout game along with a Q-learning
algorithm to train an agent for playing the game. The main components are:

1. Breakout Environment:
   - A simplified discrete grid version of Breakout with a 10x10 grid.
   - The paddle is fixed at the bottom and can move left or right.
   - Bricks are only present in the top row (represented using a bit mask).
   - The ball moves with a fixed velocity, and its direction changes on collisions with walls, bricks, or the paddle.
   - Rewards are given for hitting bricks and the paddle, and penalties are applied on each step or when the game is lost.

2. Q-Learning Algorithm:
   - A Q-table is maintained using a dictionary mapping states to Q-values for each of the three possible actions.
   - The algorithm uses an epsilon-greedy strategy with a linearly decaying epsilon for exploration.
   - At each step, the Q-value for the state-action pair is updated using the Q-learning update rule.
   - Episode rewards and success (when all bricks are removed) are recorded for performance evaluation.

3. Demonstration:
   - A graphical demo is implemented using Pygame to visualize the agent playing the Breakout game using the learned policy.

4. Learning Curve Plotting:
   - The training performance is visualized by plotting the reward per episode and the success rate over episodes.
   - A moving average is computed to smooth the curves for clearer visualization.

The main flow of the program is to train the Q-learning agent, run a demonstration of the learned policy,
and finally plot the learning curves.
"""

import pygame
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

class BreakoutEnv:
    def __init__(self):
        self.cols = 10
        self.rows = 10
        # Paddle parameters
        self.paddle_width = 3  # Paddle covers 3 grid cells; its left position can range from 0 to 7
        self.paddle_y = self.rows - 1  # Paddle is fixed at the bottom row (row 9)
        # Brick state: bricks only in the top row (row==0), represented as a 10-bit number (initially all bricks exist)
        self.initial_bricks = (1 << 10) - 1

    def reset(self):
        # Center the paddle at the start
        self.paddle_x = (self.cols - self.paddle_width) // 2
        # Initialize ball position: one row above the paddle, at a random position within the paddle
        self.ball_x = random.randint(self.paddle_x, self.paddle_x + self.paddle_width - 1)
        self.ball_y = self.paddle_y - 1
        # Ball's initial velocity: vertical upward (-1), horizontal randomly -1 or 1
        self.vx = random.choice([-1, 1])
        self.vy = -1
        # All bricks are present initially
        self.bricks = self.initial_bricks
        self.t = 0
        return self.get_state()

    def get_state(self):
        # Simplified state: (ball_x, ball_y, vx, vy, paddle_x, brick_count)
        brick_count = bin(self.bricks).count("1")
        return (self.ball_x, self.ball_y, self.vx, self.vy, self.paddle_x, brick_count)

    def step(self, action):
        """
        Action:
          0: No movement
          1: Move paddle left by one cell
          2: Move paddle right by one cell
        Updates the paddle position (kept within boundaries) and moves the ball.
        """
        # Update paddle position based on action
        if action == 1:
            self.paddle_x = max(0, self.paddle_x - 1)
        elif action == 2:
            self.paddle_x = min(self.cols - self.paddle_width, self.paddle_x + 1)

        reward = -0.1  # Base step penalty

        # Update ball position (move one cell per step)
        next_ball_x = self.ball_x + self.vx
        next_ball_y = self.ball_y + self.vy

        # Handle collision with left/right walls
        if next_ball_x < 0:
            next_ball_x = 0
            self.vx = -self.vx
        elif next_ball_x >= self.cols:
            next_ball_x = self.cols - 1
            self.vx = -self.vx

        # Handle collision with the top wall
        if next_ball_y < 0:
            next_ball_y = 0
            self.vy = -self.vy

        # Check for brick collision (bricks are in the top row, row==0)
        if next_ball_y == 0:
            brick_bit = 1 << next_ball_x
            if self.bricks & brick_bit:
                # Brick exists; remove it and reverse vertical direction
                self.bricks = self.bricks & (~brick_bit)
                self.vy = -self.vy
                reward += 5  # Reward for hitting a brick
                next_ball_y = 0  # Keep the ball in the top row

        # Check for paddle collision (paddle is in the bottom row)
        if next_ball_y == self.paddle_y:
            if self.paddle_x <= next_ball_x < self.paddle_x + self.paddle_width:
                self.vy = -self.vy
                reward += 1  # Reward for hitting the paddle
                next_ball_y = self.paddle_y - 1

        # Update ball position and time step
        self.ball_x = next_ball_x
        self.ball_y = next_ball_y
        self.t += 1

        # Determine if the episode is done
        done = False
        if self.ball_y >= self.rows:
            done = True
        if self.bricks == 0:
            done = True

        # Final reward modification at the end of the game
        if done:
            if self.ball_y >= self.rows:
                reward = -10  # Failure
            elif self.bricks == 0:
                reward = 10   # Victory

        return self.get_state(), reward, done

def train_qlearning(num_episodes=10000, alpha=0.1, gamma=0.99, initial_epsilon=0.2):
    env = BreakoutEnv()
    Q = {}  # Q-table: maps state tuple to a list of Q-values for three actions

    def get_Q(state):
        if state not in Q:
            Q[state] = [0, 0, 0]
        return Q[state]

    # Record cumulative reward and success flag (success when all bricks are cleared) per episode
    episode_rewards = []
    episode_success = []
    total_reward_accum = 0

    for episode in range(num_episodes):
        # Linear decay of epsilon from initial_epsilon to 0
        epsilon = initial_epsilon * (1 - episode / num_episodes)
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                qvals = get_Q(state)
                action = int(np.argmax(qvals))
            next_state, reward, done = env.step(action)
            episode_reward += reward
            best_next = max(get_Q(next_state))
            current = get_Q(state)[action]
            Q[state][action] = current + alpha * (reward + gamma * best_next - current)
            state = next_state

        total_reward_accum += episode_reward
        episode_rewards.append(episode_reward)
        # Record success (1 for victory, 0 for failure)
        episode_success.append(1 if env.bricks == 0 else 0)

        if (episode + 1) % 1000 == 0:
            avg_reward = total_reward_accum / 1000.0
            print(f"Episode {episode+1}/{num_episodes} - Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")
            total_reward_accum = 0

    return Q, episode_rewards, episode_success

def demo(Q):
    pygame.init()
    GRID_SIZE = 50
    cols, rows = 10, 10
    WIDTH, HEIGHT = cols * GRID_SIZE, rows * GRID_SIZE
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Q-Learning Breakout Demo")
    clock = pygame.time.Clock()

    env = BreakoutEnv()
    state = env.reset()

    running = True
    while running:
        clock.tick(5)  # Demo speed: 5 frames per second
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Use greedy strategy based on learned Q-values
        if state in Q:
            action = int(np.argmax(Q[state]))
        else:
            action = 0

        state, reward, done = env.step(action)

        # Draw game scene
        screen.fill((255, 255, 255))
        # Draw bricks in the top row
        for i in range(10):
            if env.bricks & (1 << i):
                rect = pygame.Rect(i * GRID_SIZE, 0, GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(screen, (255, 0, 0), rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)
        # Draw paddle
        paddle_rect = pygame.Rect(env.paddle_x * GRID_SIZE, env.paddle_y * GRID_SIZE, env.paddle_width * GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, (0, 0, 255), paddle_rect)
        # Draw ball as a circle
        center = (int(env.ball_x * GRID_SIZE + GRID_SIZE / 2), int(env.ball_y * GRID_SIZE + GRID_SIZE / 2))
        radius = GRID_SIZE // 2
        pygame.draw.circle(screen, (255, 0, 0), center, radius)
        pygame.display.flip()

        if done:
            running = False

    pygame.time.wait(3000)
    pygame.quit()

def plot_learning_curves(episode_rewards, episode_success, window=100):
    episodes = np.arange(len(episode_rewards))
    rewards = np.array(episode_rewards)
    success = np.array(episode_success)
    
    # Compute moving averages using convolution
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    reward_avg = moving_average(rewards, window)
    success_rate = moving_average(success, window)
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Reward vs. Episode
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, alpha=0.3, label="Episode Reward")
    plt.plot(episodes[window-1:], reward_avg, label=f"Moving Average (w={window})", color='red')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.legend()
    
    # Subplot 2: Success Rate vs. Episode
    plt.subplot(1, 2, 2)
    plt.plot(episodes[window-1:], success_rate, label=f"Success Rate (w={window})", color='green')
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs Episode")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Training Q-learning agent...")
    Q, episode_rewards, episode_success = train_qlearning(num_episodes=100000, initial_epsilon=0.2)
    print("Training completed. Now demonstrating the learned policy...")
    demo(Q)
    print("Plotting learning curves...")
    plot_learning_curves(episode_rewards, episode_success, window=100)
