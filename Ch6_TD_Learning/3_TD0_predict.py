import random
import matplotlib.pyplot as plt

class TD0:
    def __init__(self):
        # 这里只考虑 "town" 和 "castle"，暂时不使用 "forest"
        self.V = {"town": 0.0, "castle": 0.0}  
        self.gamma = 0.95
        self.alpha = 0.05

    def move(self, S):
        """
        根据当前状态 S 返回新的状态 S_new 和奖励 R_new
        """
        p = random.random()
        if S == "town":
            # 当在 town 时，策略规定选择去 castle，
            # 成功概率为 0.1：成功时转移到 castle，并获得奖励 +8；
            # 否则转移失败，状态保持 town，并获得惩罚 -5。
            if p < 0.1:
                return "castle", +8
            else:
                return "town", -5
        elif S == "castle":
            # 当在 castle 时，策略规定选择去 town，
            # 成功概率为 0.3：成功时转移到 town，并获得奖励 +5；
            # 否则转移失败，状态保持 castle，并获得惩罚 -1。
            if p < 0.3:
                return "town", +5
            else:
                return "castle", -1

    def compute_delta(self, S, S_new, R_new):
        """
        计算 TD 误差: delta = R_new + gamma * V(S_new) - V(S)
        """
        return R_new + self.gamma * self.V[S_new] - self.V[S]

    def main(self):
        num_episodes = 10000
        learning_curve = []  # 记录每个 episode 后 V 的快照（例如 town 和 castle 的值）
        episode_rewards = []  # 每个 episode 的累计奖励
        episode_success_rate = []  # 每个 episode 的成功转移率
        
        for episode in range(num_episodes):
            S = "town"  # 每个 episode 从 "town" 开始
            total_reward = 0  # 累计奖励初始化
            success_count = 0  # 成功转移次数初始化
            
            for t in range(10):  # 每个 episode 固定 10 个 timestep
                S_new, R_new = self.move(S)
                # 判断是否成功转移：如果 S_new 与 S 不同，则说明转移成功
                if S_new != S:
                    success_count += 1
                total_reward += R_new
                
                delta = self.compute_delta(S, S_new, R_new)
                # 按 TD(0) 的增量更新公式更新当前状态的价值
                self.V[S] = self.V[S] + self.alpha * delta
                S = S_new  # 更新状态
            
            # 保存每个 episode 结束后的价值函数快照（可选）
            learning_curve.append((self.V["town"], self.V["castle"]))
            # 保存累计奖励
            episode_rewards.append(total_reward)
            # 成功率 = 成功转移次数 / 10
            episode_success_rate.append(success_count / 10.0)
        
        # 打印最终状态价值函数
        print("最终状态价值函数:")
        print("V(town):", self.V["town"])
        print("V(castle):", self.V["castle"])

        # 绘制两条学习曲线
        episodes = range(num_episodes)
        plt.figure(figsize=(12, 5))
        
        # 第一幅图：Reward vs Episode
        plt.subplot(1, 2, 1)
        plt.plot(episodes, episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Learning Curve: Reward vs Episode")
        plt.grid(True)
        
        # 第二幅图：Success Rate vs Episode
        plt.subplot(1, 2, 2)
        plt.plot(episodes, episode_success_rate)
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.title("Learning Curve: Success Rate vs Episode")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    agent = TD0()
    agent.main()
