from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

MAX_SPEED = 4

CELL_EDGE = 0
CELL_START_LINE = 2
CELL_TRACK = 1
CELL_FINISH_LINE = 3

REWARD_FINISH = 0
REWARD_MOVE = -1
REWARD_OUT_OF_TRACK = -100
REWARD_TIMEOUT = -200  # 超过最大步数时的惩罚
INITIAL_VALUE = -150

MAX_STEPS_PER_EPISODE = 10000  # 回合最大步数

class Env:
    def __init__(self, grid, pos0=None):
        self.grid = grid
        self.pos0 = pos0
        self.position = None
        self.speed = (0, 0)
        self.rows, self.cols = grid.shape

    def reset(self):
        """
        回合开始或撞墙后调用：
        - 如果 self.pos0=None，则在起始线的所有格子中随机选择一个作为位置
        - 速度置为 (0,0)
        - 返回 (position, speed)
        """
        start_line_indices = np.where(self.grid == CELL_START_LINE)
        selected_index = np.random.randint(0, len(start_line_indices[0]))

        if self.pos0 is None:
            self.position = (
                start_line_indices[0][selected_index],
                start_line_indices[1][selected_index]
            )
        else:
            self.position = self.pos0

        self.speed = (0, 0)
        return (self.position, self.speed)

    def act(self, action):
        """
        对赛车执行一次动作 (ax, ay)，更新速度和位置。
        在更新位置前，先做“插值检查”，逐步走每个“子步”。
        
        - 0.1 概率将 (ax, ay) 改为 (0, 0) [书中额外难度]
        - 新速度 speedp = clamp( speed + action, [0, MAX_SPEED] )
        - 然后从旧位置到新位置逐格移动，随时检测：
            * 若途中穿过终点线 => 回合结束 (reward=0, sp=None)
            * 若途中越界 => 传送回起点 (reward=-100, sp=(pos0, 0速度))，继续回合
            * 若都没问题 => 正常到达最终位置 (reward=-1, sp=新状态)
        """
        # 1) 以0.1概率将加速度覆盖为(0,0)
        if np.random.rand() < 0.1:
            action = (0, 0)

        # 2) 计算新速度，速度分量在 [0, MAX_SPEED] 之间
        vx_new = max(min(self.speed[0] + action[0], MAX_SPEED), 0)
        vy_new = max(min(self.speed[1] + action[1], MAX_SPEED), 0)

        # 3) 准备做插值检查
        old_r, old_c = self.position
        # 注意题目里：向上移动row要减，向右移动col要加
        # 新的目标位置:
        target_r = old_r - vx_new
        target_c = old_c + vy_new

        # 我们逐步移动 row 方向，再逐步移动 col 方向（最简单的做法）
        # row_step = -1 if vx_new>0，否则0
        row_step = -1 if vx_new > 0 else 0
        # col_step = 1 if vy_new>0，否则0
        col_step = 1 if vy_new > 0 else 0

        # 需要移动的步数:
        steps_in_row = abs(vx_new)  # 要走多少格
        steps_in_col = abs(vy_new)

        # 先走 row 方向
        r, c = old_r, old_c
        for _ in range(steps_in_row):
            r += row_step
            # 每移动 1 步都检查是否越界/终点
            # 终点或边界可能在 r,c
            check_result = self._check_cell(r, c)
            if check_result is not None:
                return check_result  # 里面会返回 (reward, new_state/sp=None)

        # 再走 col 方向
        for _ in range(steps_in_col):
            c += col_step
            check_result = self._check_cell(r, c)
            if check_result is not None:
                return check_result

        # 如果所有子步都没出界也没到终点，则更新位置、速度
        self.position = (r, c)
        self.speed = (vx_new, vy_new)
        return (REWARD_MOVE, (self.position, self.speed))

    def _check_cell(self, r, c):
        """
        检查位置 (r, c) 是否出界、是否终点。
        - 若终点 => (REWARD_FINISH, None) 表示回合结束
        - 若出界 => (REWARD_OUT_OF_TRACK, (reset后的新状态))
        - 否则 => None
        """
        # 先检查是否越界(行或列超范围)
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            # 越界
            s0 = self.reset()
            return (REWARD_OUT_OF_TRACK, s0)

        cell_type = self.grid[r, c]
        if cell_type == CELL_FINISH_LINE:
            # 到达终点 => 结束回合
            return (REWARD_FINISH, None)
        elif cell_type == CELL_EDGE:
            # 撞墙 => 回到起点
            s0 = self.reset()
            return (REWARD_OUT_OF_TRACK, s0)

        # 否则啥也不做 => None
        return None


def action_gen():
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            yield (i, j)


def play(env, policy, max_steps=MAX_STEPS_PER_EPISODE):
    """
    与环境交互，直到回合结束 (sp=None) 或超过 max_steps。
    返回 (states, actions, rewards) 便于蒙特卡洛回溯。
    """
    states = []
    actions = []
    rewards = []

    step_count = 0

    s0 = env.reset()
    a0 = policy(s0)

    states.append(s0)
    actions.append(a0)

    rp, sp = env.act(a0)
    step_count += 1

    while sp is not None and step_count < max_steps:
        rewards.append(rp)
        states.append(sp)

        ap = policy(sp)
        actions.append(ap)

        rp, sp = env.act(ap)
        step_count += 1

    if sp is None:
        # 正常结束(到终点)
        rewards.append(rp)
    else:
        # 超过 max_steps => 给一个 REWARD_TIMEOUT, 结束回合
        rewards.append(REWARD_TIMEOUT)

    return states, actions, rewards


def mc_control(grid, gamma=1.0, max_episode=10_000, env_name='env0'):
    Q = defaultdict(lambda: INITIAL_VALUE)
    C = defaultdict(float)
    pi = np.zeros((grid.shape[0], grid.shape[1], MAX_SPEED + 1, MAX_SPEED + 1), dtype=(int, 2))
    env = Env(grid)

    for _ in tqdm(range(max_episode)):
        # 使用纯随机行为策略 => off-policy
        states, actions, rewards = play(env, policy=lambda s: (np.random.randint(-1, 2), np.random.randint(-1, 2)),
                                        max_steps=MAX_STEPS_PER_EPISODE)
        G = 0
        W = 1

        # 从后往前回溯
        for st, at, rt in zip(reversed(states), reversed(actions), reversed(rewards)):
            G = gamma * G + rt
            stat = (st, at)

            C[stat] += W
            Q[stat] += (W / C[stat]) * (G - Q[stat])

            # 更新目标策略 (greedy w.r.t Q)
            action_list = list(action_gen())
            action_values = [Q[(st, a)] for a in action_list]
            best_a = action_list[np.argmax(action_values)]
            pi[st[0][0], st[0][1], st[1][0], st[1][1]] = best_a

            # off-policy重要性采样，如果行为动作 != 目标动作，则后续权重=0 => break
            if at != best_a:
                break

            # 行为策略是 1/9 => ratio = pi(a|s)/b(a|s) = 1 / (1/9) = 9
            W *= 9

    with open(env_name + '_policy.obj', 'wb') as f:
        pickle.dump(pi, f)

    V_avg = build_avg_value_function(Q, grid)
    plot_value_function(V_avg, max_episode, env_name + "_value_avg.png")

    V_max = build_max_value_function(Q, grid)
    plot_value_function(V_max, max_episode, env_name + "_value_max.png")


def build_avg_value_function(Q, grid):
    rows, cols = grid.shape
    V = np.zeros(grid.shape)
    for i in range(rows):
        for j in range(cols):
            avg = 0
            n = 0
            for si in range(MAX_SPEED + 1):
                for sj in range(MAX_SPEED + 1):
                    for ai in [-1, 0, 1]:
                        for aj in [-1, 0, 1]:
                            key = (((i, j), (si, sj)), (ai, aj))
                            if key in Q:
                                qv = Q[key]
                                n += 1
                                avg += (qv - avg) / n
            V[i, j] = avg
    return V


def build_max_value_function(Q, grid):
    rows, cols = grid.shape
    V = np.zeros(grid.shape) - 30
    for i in range(rows):
        for j in range(cols):
            values = [-30]
            for si in range(MAX_SPEED + 1):
                for sj in range(MAX_SPEED + 1):
                    for ai in [-1, 0, 1]:
                        for aj in [-1, 0, 1]:
                            key = (((i, j), (si, sj)), (ai, aj))
                            if key in Q:
                                values.append(Q[key])
            V[i, j] = max(values)
    return V


def race_track(grid, gamma=1.0, env_name='env0'):
    mc_control(grid, gamma, env_name=env_name, max_episode=50000)


def plot_value_function(V, episode_count, file_name='5_3_2.png'):
    plt.figure()
    sns.heatmap(V, cmap="YlGnBu")
    plt.title(f'Value Function ({episode_count} episodes)')
    plt.savefig(file_name)
    plt.close()


def generate_trajectory(grid, pi, pos0, env_name):
    """
    这里也要做“插值检查”，
    所以直接用和训练时同一个 Env，且同样的 act() 函数。
    我们在绘制时，把每个“子步”也标出来。
    """
    env = Env(grid, pos0)
    # 重置到指定起点
    s0 = env.reset()
    # 用一个数组 V 做可视化背景
    V = np.copy(grid)

    # 先标记起点
    V[s0[0][0], s0[0][1]] = 4

    (r, c), (vx, vy) = s0
    a0 = pi[r, c, vx, vy]
    rp, sp, sub_positions = act_with_substeps(env, a0)

    # sub_positions 里包含了每个子步的位置(含起点?)
    for (rr, cc) in sub_positions:
        V[rr, cc] = 4

    while sp is not None:
        (r, c), (vx, vy) = sp
        a = pi[r, c, vx, vy]

        rp, sp, sub_positions = act_with_substeps(env, a)
        for (rr, cc) in sub_positions:
            V[rr, cc] = 4

    plot_trajectory(V, file_name=env_name + f'_demo_e_5_12_{pos0[1]}.png')


def act_with_substeps(env, action):
    """
    和 env.act() 类似，但把“子步”路径记录下来，便于可视化。
    返回: (reward, new_state, sub_positions)
      - sub_positions: 这一步走过的所有 (row, col)，含终点/撞墙前最后一个。
    """
    # 先保存一下老位置
    old_r, old_c = env.position
    old_speed = env.speed

    # 0.1 概率将加速度覆盖为(0,0)
    actual_action = action
    if np.random.rand() < 0.1:
        actual_action = (0, 0)

    # 新速度
    vx_new = max(min(old_speed[0] + actual_action[0], MAX_SPEED), 0)
    vy_new = max(min(old_speed[1] + actual_action[1], MAX_SPEED), 0)

    sub_positions = []
    r, c = old_r, old_c

    # row_step, col_step
    row_step = -1 if vx_new > 0 else 0
    col_step = 1 if vy_new > 0 else 0

    steps_in_row = abs(vx_new)
    steps_in_col = abs(vy_new)

    # 逐步走 row
    for _ in range(steps_in_row):
        r += row_step
        # 检查
        check_result = env._check_cell(r, c)
        sub_positions.append((r, c))
        if check_result is not None:
            # 撞墙 or 终点
            reward, sp = check_result
            if sp is None:
                # 到终点 => episode over
                return (reward, None, sub_positions)
            else:
                # 撞墙 => sp = env.reset() 回到起点
                return (reward, sp, sub_positions)

    # 逐步走 col
    for _ in range(steps_in_col):
        c += col_step
        check_result = env._check_cell(r, c)
        sub_positions.append((r, c))
        if check_result is not None:
            reward, sp = check_result
            if sp is None:
                return (reward, None, sub_positions)
            else:
                return (reward, sp, sub_positions)

    # 如果没撞墙也没到终点 => 更新位置、速度
    env.position = (r, c)
    env.speed = (vx_new, vy_new)
    return (REWARD_MOVE, (env.position, env.speed), sub_positions)


def generate_trajectory_plots(grid, env_name):
    with open(env_name + '_policy.obj', 'rb') as f:
        pi = pickle.load(f)
    # 对最后一行中的起始线格子，逐一画轨迹
    last_row = grid.shape[0] - 1
    for col in range(grid.shape[1]):
        if grid[last_row, col] == CELL_START_LINE:
            generate_trajectory(grid, pi, (last_row, col), env_name)


def plot_trajectory(V, file_name='track.png'):
    plt.figure(figsize=(V.shape[1] * 0.25, V.shape[0] * 0.25))
    sns.heatmap(V, cmap="YlGnBu", cbar=False, linewidths=0.1, linecolor='gray')
    plt.title('Path', fontsize=10)
    plt.savefig(file_name)
    plt.close()


# -------------------- 示例的两个 grid --------------------
grid1_str = """0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 0 2 2 2 2 2 2 0 0 0 0 0 0 0 0
"""

grid2_str = """0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0
"""

grid1 = np.array([[int(num) for num in line.split()] for line in grid1_str.strip().splitlines()])
grid2 = np.array([[int(num) for num in line.split()] for line in grid2_str.strip().splitlines()])


if __name__ == '__main__':
    # 训练并生成轨迹图 (grid1)
    race_track(grid1, gamma=1.0, env_name='grid1')
    generate_trajectory_plots(grid1, env_name='grid1')

    # 训练并生成轨迹图 (grid2)
    race_track(grid2, gamma=1.0, env_name='grid2')
    generate_trajectory_plots(grid2, env_name='grid2')
