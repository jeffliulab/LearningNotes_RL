import random
import matplotlib.pyplot as plt

class Sarsa:
    def __init__(self):
        # 设置训练参数
        self.EPISODES = 10000                          # 总共训练的episode数量
        self.EPISODE_TIME_STEPS = 10                   # 每个episode中只有10个timestep
        self.gamma = 0.95                              # 折扣因子
        self.alpha = 0.1                               # 学习率
        self.epsilon = 0.1                             # e-greedy中的epsilon值（探索概率）

        # 状态空间：town, forest, castle
        self.states = ['town', 'forest', 'castle']

        # Action：在每个地方，必须选择另外两个地方中的一个作为该回合的action
        # 例如，在town状态下，允许的动作为去forest或castle
        self.actions = {
            'town': ['forest', 'castle'],
            'forest': ['town', 'castle'],
            'castle': ['town', 'forest']
        }

        # 定义各场景的奖励和惩罚（Reward）： (成功时奖励, 失败时惩罚)
        # town: +5 / -1, castle: +8 / -5, forest: +2 / -3
        self.reward_structure = {
            'town': (5, -1),
            'castle': (8, -5),
            'forest': (2, -3)
        }

        # 各场景之间的转移成功率： P (S' | S, a)
        # 如果转移成功，则进行状态转移并获得目标场景的奖励；
        # 如果失败，则状态不改变，并获得目标场景的惩罚
        self.success_prob = {
            ('town', 'castle'): 0.1,
            ('town', 'forest'): 0.7,
            ('forest', 'castle'): 0.1,
            ('forest', 'town'): 0.8,
            ('castle', 'town'): 0.3,
            ('castle', 'forest'): 0.8
        }

        # 初始化Q函数，Q值存储为字典，键为(state, action)对，初始值均设为0
        self.Q = {}
        for state in self.states:
            for action in self.actions[state]:
                self.Q[(state, action)] = 0.0

        # 用于记录每个episode的累计奖励和成功率（成功次数/总次数）
        self.episode_rewards = []
        self.episode_success_rates = []

    def action(self, state):
        """
        根据e-greedy策略来选取action
        e-greedy的epsilon：以epsilon的概率随机选择一个动作进行探索，
        以(1-epsilon)的概率选择当前看来最优的动作进行利用。
        """
        possible_actions = self.actions[state]
        if random.random() < self.epsilon:
            # 探索：随机选择一个动作
            return random.choice(possible_actions)
        else:
            # 利用：选择Q值最高的动作
            q_values = [self.Q[(state, a)] for a in possible_actions]
            max_q = max(q_values)
            # 如果有多个动作具有相同的最大Q值，则随机选择其中之一
            best_actions = [a for a in possible_actions if self.Q[(state, a)] == max_q]
            return random.choice(best_actions)

    def get_reward(self, current_state, action):
        """
        根据当前状态和对应的动作，获得奖励R和下一个状态S
        转移成功的概率根据self.success_prob中的定义，
        如果转移成功，则状态转移到目标状态，并获得该场景的奖励；
        如果转移失败，则状态不改变，并获得目标场景的惩罚。
        返回：(reward, next_state, success_flag)
        """
        target_state = action  # 动作即为目标状态
        # 获取转移成功的概率
        success_probability = self.success_prob.get((current_state, target_state), 0)
        if random.random() < success_probability:
            # 转移成功：状态变为目标状态，获得正向奖励
            reward = self.reward_structure[target_state][0]
            next_state = target_state
            success = True
        else:
            # 转移失败：状态保持不变，获得目标状态对应的惩罚
            reward = self.reward_structure[target_state][1]
            next_state = current_state
            success = False
        return reward, next_state, success

    def sarsa(self):
        """
        用SARSA算法进行学习：
        - 每个episode有固定的10个timestep，初始状态为town
        - 每个timestep中：
            1. 根据当前状态选取动作（e-greedy策略）
            2. 执行动作获得奖励和下一个状态
            3. 在下一个状态中选取下一动作
            4. 使用更新公式更新Q值：
               Q(S_t, A_t) = Q(S_t, A_t) + alpha * (R_{t+1} + gamma * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))
        - 同时记录每个episode的累计奖励和成功率（成功次数/总次数）
        """
        for episode in range(self.EPISODES):
            current_state = 'town'             # 初始状态为town
            current_action = self.action(current_state)  # 根据当前状态选取初始动作
            episode_reward = 0                 # 当前episode的累计奖励， 这里计算这个数值的目的是绘制学习曲线，而不参与sarsa迭代
            success_count = 0                  # 当前episode的成功转移次数

            for t in range(self.EPISODE_TIME_STEPS):
                # 根据当前状态和动作获得奖励以及下一个状态，注意还返回是否转移成功
                reward, next_state, success = self.get_reward(current_state, current_action)
                episode_reward += reward
                if success:
                    success_count += 1

                # 在下一个状态中选择下一动作
                next_action = self.action(next_state)

                # 更新Q值：使用SARSA的更新公式
                self.Q[(current_state, current_action)] += self.alpha * (
                    reward + self.gamma * self.Q[(next_state, next_action)] - self.Q[(current_state, current_action)]
                )

                # 更新状态和动作
                current_state = next_state
                current_action = next_action

            # 记录本episode的累计奖励和成功率
            self.episode_rewards.append(episode_reward)
            self.episode_success_rates.append(success_count / self.EPISODE_TIME_STEPS)

            # 可选：每隔一定episode输出一下当前进度
            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode+1}/{self.EPISODES}, Total Reward: {episode_reward}, Success Rate: {self.episode_success_rates[-1]:.2f}")

    def main(self):
        """
        主函数：
        1. 运行SARSA算法
        2. 输出最终的Q值
        3. 绘制学习曲线：奖励 vs Episode 和 成功率 vs Episode
        """
        self.sarsa()

        # 输出最终的Q值
        print("Final Q-values:")
        for (state, action), value in self.Q.items():
            print(f"State {state}, Action {action}: {value:.2f}")

        # 绘制学习曲线：奖励 vs Episode
        plt.figure()
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Learning Curve: Reward vs Episode")
        plt.show()

        # 绘制学习曲线：成功率 vs Episode
        plt.figure()
        plt.plot(self.episode_success_rates)
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.title("Learning Curve: Success Rate vs Episode")
        plt.show()


if __name__ == '__main__':
    agent = Sarsa()
    agent.main()
