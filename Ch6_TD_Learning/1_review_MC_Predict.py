# 勇者游戏
# 
# 为了更好复习MC/TD，我特别设计了一个简单的小游戏：
# 在地图上，有town, forest, castle三个地方，代表三个状态
# 勇者（agent）总是出生在town
# 每个回合中，勇者必须进行10次游戏，每次游戏是一个timestep
# 每个timestep中，勇者必须前往另一个场景；比如勇者在town时，他必须选择前往castle或者forest
# 在勇者从一个地方到另一个地方的时候，有可能成功，也有可能失败，即状态转移概率
# 如果转移成功，则获得对应场景的奖励；如果失败，则获得对应场景的惩罚
#
# 各场景的奖励和惩罚分别如下：（Reward）
#    town: +5 / -1
#    castle: +8 / -5
#    forest: +2 / -3
#
# 各场景之间的转移成功率如下： P (S' | S, a)
# 该概率是独立事件
# 如果转移成功，则进行状态转移；如果失败，则状态不改变，并获得目标场景的惩罚
#    town to castle: 0.1
#    town to forest: 0.7
#    forest to castle: 0.1
#    forest to town: 0.8
#    castle to town: 0.3
#    castle to forest: 0.8
#
# Action：在每个地方，必须选择另外两个地方中的一个作为该回合的action
# Policy：如果有具体的策略，则可进行prediction (policy evaluation)
# Prediction problem：首先设定一个策略，进行predict
# 折扣因子：可以设定一个0.95
#
###########################################################
#
# 游戏环境设置：
# 策略：不在城堡时，选择去城堡；否则选择去town
# 目标：进行10000个episodes，每个episode内固定10个timestep，评估该策略的Value Function

# 初始化：
#     state状态空间：town, forest, castle
#
#     初始化一个空的list，list_V, 存储（V town, V castle, V forest）
#
#     全局变量V存储 V town, V forest, V castle
#
#     gamma = 0.95

import matplotlib.pyplot as plt
import random
random.seed(42)

class MC:
    def __init__(self):
        # state = "town" # 初始化状态为town
        self.states = ["town", "forest", "castle"] # 定义状态空间
        # v_town = 0
        # v_forest = 0
        # v_castle = 0
        self.V = {"town": 0.0, "forest": 0.0, "castle": 0.0}
        self.list_V = []
        self.gamma = 0.95
        self.alpha = 0.01
        self.FIXED_TIMESTEP = 100 # 把每个episode的固定的timestep数写在这里
        self.EPISODES = 100000

    def policy(self, current_state):
        # 不设置全局state变量是为了防止episode之间互相干扰
        """
        当前策略： 如果不在castle就去castle, 否则去town
        """
        if current_state in ["town", "forest"]:
            return "castle"
        else:
            return "town"
        
    def state_update(self, current_state, action): # 更新状态并返回reward
        """
        根据当前状态和选定动作（目标状态），使用给定的转移概率更新状态，并返回奖励。
        注意：如果转移失败，则状态保持不变，但奖励为目标状态对应的惩罚。
        """
        # town to castle:
        #     0.1 success: reward +8, state 转移为caslte
        #     else: reward -5
        # town to forest: 目前的策略中不存在这个选项，不过可以写下来待用
        #     0.7 success: reward +2, state 转移为forest
        #     else: reward -3
        # forest to castle: 
        #     0.1 success: reward +8, state 转移为caslte
        #     else: reward -5
        # forest to town: 目前策略中不存在这个，暂时写下来：
        #     0.8 success reward +5, state转移为town
        #     else reward -1
        # castle to town: 
        #     0.3 success, reward +5, state转移为town
        #     else reward -1
        # castle to forest: 目前策略中不存在这个
        #     0.8 success, reward +2, state 转移为forest
        #     else reward -3
        
        # 模拟随机事件
        p = random.random()  # [0,1)均匀分布
        
        # 针对每个转移写出逻辑：
        if current_state == "town" and action == "castle":
            if p < 0.1:
                new_state = "castle"
                reward = 8    # 转移成功：castle奖励
            else:
                new_state = current_state  # 转移失败，状态不变
                reward = -5   # 惩罚：castle惩罚
            return new_state, reward

        elif current_state == "town" and action == "forest":
            # 备用
            if p < 0.7:
                new_state = "forest"
                reward = 2    # 成功：forest奖励
            else:
                new_state = current_state
                reward = -3   # 惩罚：forest惩罚
            return new_state, reward

        elif current_state == "forest" and action == "castle":
            if p < 0.1:
                new_state = "castle"
                reward = 8
            else:
                new_state = current_state
                reward = -5
            return new_state, reward

        elif current_state == "forest" and action == "town":
            # 备用
            if p < 0.8:
                new_state = "town"
                reward = 5
            else:
                new_state = current_state
                reward = -1
            return new_state, reward

        elif current_state == "castle" and action == "town":
            if p < 0.3:
                new_state = "town"
                reward = 5
            else:
                new_state = current_state
                reward = -1
            return new_state, reward

        elif current_state == "castle" and action == "forest":
            # 备用
            if p < 0.8:
                new_state = "forest"
                reward = 2
            else:
                new_state = current_state
                reward = -3
            return new_state, reward



    def compute_returns(self, rewards):
        """
        计算一个episode中每个timestep的累计折扣回报 G_t。
        例如：如果 rewards = [r0, r1, ..., r_(T-1)]，则
              G_t = r_t + gamma*r_(t+1) + gamma^2*r_(t+2) + ... + gamma^(T-t-1)*r_(T-1)
        使用从后向前的递归计算方法。
        公式： Gt = Rt + gamma * Gt+1
        """
        T = len(rewards)
        returns = [0.0] * T
        cumulative = 0.0
        for t in reversed(range(T)):
            cumulative = rewards[t] + self.gamma * cumulative
            returns[t] = cumulative
        return returns

    def update_value(self, state, G_t):
        """
        使用增量更新公式更新全局状态价值：
            V(s) = V(s) + alpha * (G_t - V(s))
        """
        self.V[state] = self.V[state] + self.alpha * (G_t - self.V[state])

    def train_episode(self):
        """
        训练一个episode：
        1. 初始状态设为'town'
        2. 固定10个timestep内，根据策略选择动作，执行状态转移，记录状态和奖励
        3. 计算每个timestep的累计回报 G_t
        4. 对每个时刻对应的状态，用相应的 G_t 更新全局 V
        5. 将当前全局 V 的快照保存到 list_V 中
        """
        current_state = "town"  # episode起始状态
        rewards = []     # 记录每个timestep的奖励
        state_list = []  # 记录每个timestep对应的状态（即采取动作前的状态）
        
        # 采样轨迹：固定10个timestep
        for t in range(self.FIXED_TIMESTEP):
            state_list.append(current_state)
            action = self.policy(current_state)  # 根据当前状态选择动作（目标状态）
            # 执行状态更新，获得新状态和奖励
            new_state, reward = self.state_update(current_state, action)
            rewards.append(reward)
            current_state = new_state  # 更新当前状态

        # 计算每个timestep的累计回报 G_t
        returns = self.compute_returns(rewards)
        
        # 对每个timestep，根据其对应状态更新全局 V
        for t in range(self.FIXED_TIMESTEP):
            self.update_value(state_list[t], returns[t])
        
        # 保存当前全局 V 的快照，用于后续分析和学习曲线绘制
        self.list_V.append((self.V["town"], self.V["forest"], self.V["castle"]))

    def plot_learning_curve(self, list_V, max_points=1000):
        """
        绘制每个状态V值随episode变化的学习曲线，并对数据点进行下采样，
        以避免当episode数量非常大时，绘图数据点过多导致图形“爆炸”。

        参数：
            list_V: 一个列表，其中每个元素为一个tuple (V_town, V_forest, V_castle)，
                    表示在某个episode结束时的状态价值函数快照。
            max_points: 绘图时最多显示的点数（默认1000）。
        """
        import numpy as np
        n_points = len(list_V)
        if n_points > max_points:
            # 生成max_points个均匀间隔的索引
            indices = np.linspace(0, n_points - 1, max_points, dtype=int)
            episodes = indices
            V_town = [list_V[i][0] for i in indices]
            V_forest = [list_V[i][1] for i in indices]
            V_castle = [list_V[i][2] for i in indices]
        else:
            episodes = range(n_points)
            V_town = [v[0] for v in list_V]
            V_forest = [v[1] for v in list_V]
            V_castle = [v[2] for v in list_V]

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, V_town, label="V(town)")
        plt.plot(episodes, V_forest, label="V(forest)")
        plt.plot(episodes, V_castle, label="V(castle)")
        plt.xlabel("Episode")
        plt.ylabel("State Value")
        plt.title("Learning Curve: State Value vs Episode")
        plt.legend()
        plt.grid(True)
        plt.show()


    def main(self):
        for _ in range(self.EPISODES):
            self.train_episode()
        
        # 输出最终的状态价值函数
        print("最终状态价值函数:")
        print("V(town):", self.V["town"])
        print("V(forest):", self.V["forest"])
        print("V(castle):", self.V["castle"])
        
        # list_V中保存了每个episode结束时的快照，可用于绘制学习曲线
        # 例如：绘制每个状态的V随episode变化的曲线
        self.plot_learning_curve(self.list_V)

if __name__ == "__main__":
    mc_agent = MC()
    mc_agent.main()

# 通过这个例子复习了一下MC prediction：
# （1） Prodiction Problem 策略评估
# 在给定一个具体的策略下，求出该策略下每个状态的价值函数 V_pi(S)
# 在勇者游戏中，这个V就是最终的V。
# （2） Control Problem 控制问题（本代码中暂未实现）
# 寻找一个最优策略pi*, 使得每个状态下都能获得最大的预期累计奖励
# 即求出最优的状态价值函数V*(s), 或最优动作价值函数Q*(s,a)   