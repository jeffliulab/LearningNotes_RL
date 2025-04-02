import pygame
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 离散版 Breakout 环境定义（简化状态与奖励设计）
# ------------------------------

class BreakoutEnv:
    def __init__(self):
        # 网格大小
        self.cols = 10
        self.rows = 10
        # 挡板参数
        self.paddle_width = 3  # 挡板宽度：3 格，挡板左侧位置范围 0~(10-3)=7
        self.paddle_y = self.rows - 1  # 挡板固定在最底行（行号 9）
        # 砖块状态：只有顶行（row==0）有砖块，共 10 个，用一个10位二进制表示，初始值为 2^10 - 1 = 1023
        self.initial_bricks = (1 << 10) - 1

    def reset(self):
        # 挡板初始居中
        self.paddle_x = (self.cols - self.paddle_width) // 2
        # 球初始位置：位于挡板上方一格内，横向随机在挡板范围内
        self.ball_x = random.randint(self.paddle_x, self.paddle_x + self.paddle_width - 1)
        self.ball_y = self.paddle_y - 1
        # 球初始速度：竖直向上 (-1)，水平随机取 -1 或 1
        self.vx = random.choice([-1, 1])
        self.vy = -1
        # 砖块状态：10个砖块全存在
        self.bricks = self.initial_bricks
        self.t = 0
        return self.get_state()

    def get_state(self):
        # 简化状态： (ball_x, ball_y, vx, vy, paddle_x, brick_count)
        brick_count = bin(self.bricks).count("1")
        return (self.ball_x, self.ball_y, self.vx, self.vy, self.paddle_x, brick_count)

    def step(self, action):
        """
        动作：
          0: 不动
          1: 向左移动一格
          2: 向右移动一格
        更新挡板位置（保持在边界内）
        """
        # 更新挡板
        if action == 1:
            self.paddle_x = max(0, self.paddle_x - 1)
        elif action == 2:
            self.paddle_x = min(self.cols - self.paddle_width, self.paddle_x + 1)

        reward = -0.1  # 每步基础惩罚

        # 更新球的位置（每步移动1格）
        next_ball_x = self.ball_x + self.vx
        next_ball_y = self.ball_y + self.vy

        # 处理左右墙壁碰撞
        if next_ball_x < 0:
            next_ball_x = 0
            self.vx = -self.vx
        elif next_ball_x >= self.cols:
            next_ball_x = self.cols - 1
            self.vx = -self.vx

        # 处理顶部碰撞
        if next_ball_y < 0:
            next_ball_y = 0
            self.vy = -self.vy

        # 检查砖块碰撞（砖块在第一行，row==0）
        if next_ball_y == 0:
            brick_bit = 1 << next_ball_x
            if self.bricks & brick_bit:
                # 砖块存在，撞击砖块，消除该砖块
                self.bricks = self.bricks & (~brick_bit)
                self.vy = -self.vy
                reward += 5  # 撞砖块奖励
                next_ball_y = 0  # 保持在顶行

        # 检查挡板碰撞：挡板在最后一行（row 9）
        if next_ball_y == self.paddle_y:
            if self.paddle_x <= next_ball_x < self.paddle_x + self.paddle_width:
                self.vy = -self.vy
                reward += 1  # 成功接到球奖励
                next_ball_y = self.paddle_y - 1

        # 更新球位置
        self.ball_x = next_ball_x
        self.ball_y = next_ball_y
        self.t += 1

        # 判断终止条件
        done = False
        if self.ball_y >= self.rows:
            done = True
        if self.bricks == 0:
            done = True

        # 如果游戏结束，修改最终奖励：
        if done:
            if self.ball_y >= self.rows:
                reward = -10  # 失败
            elif self.bricks == 0:
                reward = 10   # 胜利

        return self.get_state(), reward, done

# ------------------------------
# Q-learning 算法实现
# ------------------------------

def train_qlearning(num_episodes=10000, alpha=0.1, gamma=0.99, initial_epsilon=0.2):
    env = BreakoutEnv()
    Q = {}  # Q表：key 为状态元组，value 为三个动作的 Q 值列表
    
    def get_Q(state):
        if state not in Q:
            Q[state] = [0, 0, 0]
        return Q[state]

    # 用于记录每个 episode 的累计 reward 和成功标志（成功：砖块全消除）
    episode_rewards = []
    episode_success = []  # 1 表示胜利（砖块全消除），0 表示失败

    total_reward_accum = 0

    for episode in range(num_episodes):
        # 线性衰减 epsilon：从 initial_epsilon 下降到 0
        epsilon = initial_epsilon * (1 - episode / num_episodes)
        
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # epsilon-greedy 策略
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
        # 记录成功情况：砖块全消除即胜利
        if env.bricks == 0:
            episode_success.append(1)
        else:
            episode_success.append(0)
        if (episode + 1) % 1000 == 0:
            avg_reward = total_reward_accum / 1000.0
            print(f"Episode {episode+1}/{num_episodes} - Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")
            total_reward_accum = 0
    return Q, episode_rewards, episode_success

# ------------------------------
# 演示（带图形界面）使用训练得到的策略
# ------------------------------

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
        clock.tick(5)  # 演示速度：每秒 5 帧
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 采用贪婪策略选择动作
        if state in Q:
            action = int(np.argmax(Q[state]))
        else:
            action = 0

        state, reward, done = env.step(action)

        # 绘制游戏场景
        screen.fill((255, 255, 255))
        # 绘制砖块（第一行）
        for i in range(10):
            if env.bricks & (1 << i):
                rect = pygame.Rect(i * GRID_SIZE, 0, GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(screen, (255, 0, 0), rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)
        # 绘制挡板
        paddle_rect = pygame.Rect(env.paddle_x * GRID_SIZE, env.paddle_y * GRID_SIZE, env.paddle_width * GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, (0, 0, 255), paddle_rect)
        # 绘制球（圆形）
        center = (int(env.ball_x * GRID_SIZE + GRID_SIZE/2), int(env.ball_y * GRID_SIZE + GRID_SIZE/2))
        radius = GRID_SIZE // 2
        pygame.draw.circle(screen, (255, 0, 0), center, radius)
        pygame.display.flip()

        if done:
            running = False

    pygame.time.wait(3000)
    pygame.quit()

# ------------------------------
# 绘制学习曲线
# ------------------------------

def plot_learning_curves(episode_rewards, episode_success, window=100):
    episodes = np.arange(len(episode_rewards))
    rewards = np.array(episode_rewards)
    success = np.array(episode_success)
    
    # 计算滑动窗口平均
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    reward_avg = moving_average(rewards, window)
    success_rate = moving_average(success, window)
    
    plt.figure(figsize=(12, 5))
    
    # 子图1：reward vs episode
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, alpha=0.3, label="Episode Reward")
    plt.plot(episodes[window-1:], reward_avg, label=f"Moving Average (w={window})", color='red')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.legend()
    
    # 子图2：success rate vs episode
    plt.subplot(1, 2, 2)
    plt.plot(episodes[window-1:], success_rate, label=f"Success Rate (w={window})", color='green')
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs Episode")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ------------------------------
# 主流程：先训练，再演示，再绘制学习曲线
# ------------------------------

if __name__ == "__main__":
    print("Training Q-learning agent...")
    Q, episode_rewards, episode_success = train_qlearning(num_episodes=100000, initial_epsilon=0.2)
    print("Training completed. Now demonstrating the learned policy...")
    demo(Q)
    print("Plotting learning curves...")
    plot_learning_curves(episode_rewards, episode_success, window=100)
