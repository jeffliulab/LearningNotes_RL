"""
本代码基于 Sutton & Barto 《Reinforcement Learning: An Introduction》 第 5 章示例 5.3 的思路，
实现了一个“抽象化”的 Blackjack 环境（无限牌堆）并使用 Monte Carlo ES (Exploring Starts) 来学习最优策略。
不包含任何“硬编码”或“手动指定”的策略规则，完全依靠蒙特卡洛方法在足够多的训练幕 (episodes) 后自动收敛。

最新修改：
1) 使用 Monte Carlo ES + \epsilon-greedy 后续决策来学习 Blackjack 的最优策略；
2) 训练足够多episodes (默认 5_000_000)，并固定随机种子以提高可复现性；

核心要点：
1. 状态表示：s = (player_sum, dealer_upcard, usable_ace)
   - player_sum ∈ [12..21]：低于 12 的情况不纳入状态空间，因为玩家一定会继续要牌。
   - dealer_upcard ∈ [1..10]：庄家的明牌（1 表示 A）。
   - usable_ace ∈ {True, False}：玩家是否有可用的 A（可以当作 11 而不爆）。

2. 动作 (Action)：
   - 0 = Stick（停牌）
   - 1 = Hit（要牌）

3. Exploring Starts：
   - 随机在所有可能的 (state, action) 中选一个初始 (s, a)，然后按照当前 Q 的贪心策略执行后续动作。
   - 这样可确保对每个 (s, a) 都有非零概率被采样。

4. 环境逻辑：
   - 若玩家选择 Hit，则从无限牌堆抽牌（1..9 各 1/13，10 占 4/13），更新玩家点数和“usable_ace” 状态。
     如果爆牌 (>21)，则玩家立即输 (reward = -1)。
   - 若玩家选择 Stick，则庄家开始拿牌 (dealer_play)，一直抽到点数 ≥17 或爆，然后与玩家点数比大小决定胜负 (+1/-1/0)。

5. 训练与收敛：
   - 我们在默认配置下运行 2,000,000 幕 (episodes)，确保 Q(s,a) 及策略有足够的采样来收敛。
   - 在完全一致的环境设置下（无限牌堆、庄家<17必须要牌、玩家无加倍/分牌等），该方法在足够多次模拟后会收敛到与书中图 5.2 几乎相同的最优策略和价值函数。
   - 若想进一步平滑或加快收敛，可再增加幕数或在后续使用小 ε 的 ε-greedy 策略，但本例严格依照“纯”ES（后续贪心）。

6. 可视化：
   - 绘制 2×2 布局的可视化图：
   - (row=0) usable ace，(row=1) no usable ace
   - (col=0) 策略 (policy)，(col=1) 价值函数 (value)
   - 策略图中，用色块区分 0=Stick 与 1=Hit；价值图中，用颜色梯度展示 V(s) 的高低。该设置在收敛后与书中示例几乎相同。

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from tqdm import trange

# 固定随机种子，便于复现
SEED = 42
np.random.seed(SEED)

STICK = 0
HIT = 1
ACTIONS = [STICK, HIT]

def draw_card():
    """
    从无限牌堆抽一张牌 (1..9 各占 1/13, 10 占 4/13)。
    """
    c = np.random.randint(1, 14)  # 1..13
    return min(c, 10)

def hand_value(sum_without_ace, ace_count):
    """
    给定 sum_without_ace（非A牌面总和）和 ace_count（A张数，先当1点），
    若有 A 并加10 不会爆，则记为可用A，总点数+10。
    返回 (点数, usable_ace)
    """
    total = sum_without_ace + ace_count
    usable = False
    if ace_count > 0 and total + 10 <= 21:
        total += 10
        usable = True
    return total, usable

def player_hit(player_sum, usable_ace):
    """
    玩家要牌。返回 (新点数, 新usable_ace, 是否爆牌)。
    """
    card = draw_card()
    # 把 (player_sum, usable_ace) 还原到 (sum_without_ace, ace_count)
    if usable_ace:
        ace_count = 1
        sum_without_ace = player_sum - 11
    else:
        ace_count = 0
        sum_without_ace = player_sum

    if card == 1:
        ace_count += 1
    else:
        sum_without_ace += card

    new_sum, new_usable = hand_value(sum_without_ace, ace_count)
    bust = (new_sum > 21)
    return new_sum, new_usable, bust

def dealer_play(dealer_upcard):
    """
    庄家先抽暗牌，再不断要牌至 >=17 或爆。
    返回 (庄家点数, usable_ace)。
    """
    second = draw_card()
    sum_without_ace = 0
    ace_count = 0

    # 明牌
    if dealer_upcard == 1:
        ace_count += 1
    else:
        sum_without_ace += dealer_upcard

    # 暗牌
    if second == 1:
        ace_count += 1
    else:
        sum_without_ace += second

    dealer_sum, dealer_usable = hand_value(sum_without_ace, ace_count)
    while dealer_sum < 17:
        c = draw_card()
        if c == 1:
            ace_count += 1
        else:
            sum_without_ace += c
        dealer_sum, dealer_usable = hand_value(sum_without_ace, ace_count)
        if dealer_sum > 21:
            break

    return dealer_sum, dealer_usable

def step(state, action):
    """
    环境一步：
    state = (player_sum, dealer_up, usable_ace)
    action = 0(STICK) or 1(HIT)
    返回 (next_state, reward, done)
    """
    player_sum, dealer_up, player_usable = state

    if action == HIT:
        new_sum, new_usable, bust = player_hit(player_sum, player_usable)
        if bust:
            return None, -1, True
        else:
            return (new_sum, dealer_up, new_usable), 0, False
    else:
        # STICK => 庄家回合
        dealer_sum, _ = dealer_play(dealer_up)
        if dealer_sum > 21:
            return None, +1, True
        else:
            if player_sum > dealer_sum:
                return None, +1, True
            elif player_sum < dealer_sum:
                return None, -1, True
            else:
                return None, 0, True

def random_state_action():
    """
    Exploring Starts: 在所有 [12..21]×[1..10]×{False,True} 中随机选 (s,a)。
    """
    player_sum = np.random.randint(12, 22)
    dealer_up = np.random.randint(1, 11)
    ace = np.random.choice([False, True])
    action = np.random.choice(ACTIONS)
    return (player_sum, dealer_up, ace), action

def generate_episode_es_epsilon(Q, epsilon=0.05):
    """
    生成一个 episode：
    1) Exploring Starts: 初始 (s,a) 随机
    2) 之后采用 epsilon-greedy(Q) 决策，直到终止
    返回 [(s0,a0,r1), (s1,a1,r2), ...]
    """
    episode = []
    state, action = random_state_action()
    done = False

    while True:
        next_state, reward, done = step(state, action)
        episode.append((state, action, reward))
        if done:
            break

        # 后续动作用 epsilon-greedy(Q)
        state = next_state
        if np.random.rand() < epsilon:
            action = np.random.choice(ACTIONS)
        else:
            action = np.argmax(Q[state])

    return episode

def mc_es_epsilon_blackjack(num_episodes=5_000_000, epsilon=0.05):
    """
    MC ES + epsilon-greedy 后续决策，
    幕数默认 5_000_000。
    """
    Q = defaultdict(lambda: np.zeros(2))
    returns = defaultdict(list)

    for _ in trange(num_episodes, desc="Training"):
        episode = generate_episode_es_epsilon(Q, epsilon)
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t = episode[t]
            G += r_t
            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                returns[(s_t, a_t)].append(G)
                Q[s_t][a_t] = np.mean(returns[(s_t, a_t)])

    policy = {}
    for s, q_vals in Q.items():
        policy[s] = np.argmax(q_vals)

    return Q, policy

def get_value_function(Q):
    """
    V(s) = max_a Q(s,a)
    """
    V = {}
    for s, q_vals in Q.items():
        V[s] = np.max(q_vals)
    return V

# ---------- 2×2 可视化 ----------
def plot_policy_value_2x2(policy, V):
    """
    画一个 2×2 图：
      上行: usable ace
        左: 策略 (policy)    右: 价值函数 (value)
      下行: no usable ace
        左: 策略 (policy)    右: 价值函数 (value)

    策略图: 用 color map 表示 0=Stick, 1=Hit
    价值图: 用渐变色(如 'viridis')显示 V(s) 大小
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 两行 (usable_ace=True / False)，两列 (policy / value)
    for row_idx, ace in enumerate([True, False]):
        # --- 先画策略 (col=0) ---
        ax_policy = axes[row_idx, 0]
        x_vals = range(1, 11)   # dealer up
        y_vals = range(12, 22)  # player sum
        policy_grid = np.zeros((len(y_vals), len(x_vals)), dtype=float)

        for i, d_up in enumerate(x_vals):
            for j, p_sum in enumerate(y_vals):
                s = (p_sum, d_up, ace)
                policy_grid[j, i] = policy.get(s, 0)

        # 我们可用一个离散的 cmap，如 'Blues'，0 和 1 两个色阶
        # 或者 'binary'，但为了对比明显，可以尝试 'coolwarm'、'cividis' 等
        im1 = ax_policy.imshow(
            policy_grid,
            origin='lower',
            extent=[1, 10, 12, 21],
            cmap=plt.cm.Blues,
            vmin=0, vmax=1,  # 明确 0 和 1 的范围
            aspect='auto'
        )
        ax_policy.set_xlabel("Dealer showing")
        ax_policy.set_ylabel("Player sum")
        ax_policy.set_title(f"Policy (usable_ace={ace})\n0=Stick, 1=Hit")

        # --- 再画价值 (col=1) ---
        ax_value = axes[row_idx, 1]
        value_grid = np.zeros((len(y_vals), len(x_vals)), dtype=float)
        for i, d_up in enumerate(x_vals):
            for j, p_sum in enumerate(y_vals):
                s = (p_sum, d_up, ace)
                value_grid[j, i] = V.get(s, 0.0)

        # 这里用 'viridis' 显示价值大小
        im2 = ax_value.imshow(
            value_grid,
            origin='lower',
            extent=[1, 10, 12, 21],
            cmap='viridis',
            aspect='auto'
        )
        ax_value.set_xlabel("Dealer showing")
        ax_value.set_ylabel("Player sum")
        ax_value.set_title(f"Value (usable_ace={ace})")

        # 可加 colorbar
        fig.colorbar(im1, ax=ax_policy, fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax_value, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# ---------- 主程序 ----------
if __name__ == "__main__":
    # 1) 训练 5,000,000 幕，epsilon=0.05
    Q, policy = mc_es_epsilon_blackjack(num_episodes=5_000_000, epsilon=0.05)
    V = get_value_function(Q)

    # 2) 2×2 绘图
    plot_policy_value_2x2(policy, V)
