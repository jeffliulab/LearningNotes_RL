import os
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from tqdm import tqdm

# -------------------- Global Parameter Settings -------------------- #

EPISODES = 50000              # Maximum number of training episodes
GAMMA = 1.0                   # Discount factor
MAX_STEPS_PER_EPISODE = 10000 # Maximum steps per episode
MAX_TRAJECTORY_STEPS = 2000   # Maximum steps for trajectory visualization
VELOCITY_LIMIT = 4            # Maximum velocity component

# Map cell types
EDGE = 'E'         # Edge cell
TRACK_LINE = '0'   # Drivable track cell
START_LINE = 'S'   # Start line cell
FINISH_LINE = 'F'  # Finish line cell

# Reward settings
REWARD_FINISH = 0
REWARD_MOVE = -1
REWARD_OUT_OF_TRACK = -100
REWARD_TIMEOUT = -200
INITIAL_VALUE = -150

# Create a results folder named by current date (YYYYMMDD)
RESULTS_FOLDER = datetime.datetime.now().strftime("%Y%m%d")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# -------------------- RaceTrack Environment -------------------- #

class RaceTrack:
    """
    RaceTrack simulates a discrete grid environment for a racing task.
    Each cell is one of: 'E', '0', 'S', 'F'.
    """
    def __init__(self, track_map, initial_position=None):
        self.track_map = track_map
        self.initial_position = initial_position
        self.current_position = None
        self.current_velocity = (0, 0)
        self.rows, self.cols = track_map.shape

    def reset(self):
        """
        Resets the environment:
        If initial_position is None, a random start cell is selected.
        Velocity is set to (0,0).
        Returns (position, velocity).
        """
        start_indices = np.where(self.track_map == START_LINE)
        idx = np.random.randint(0, len(start_indices[0]))

        if self.initial_position is None:
            row = start_indices[0][idx]
            col = start_indices[1][idx]
            self.current_position = (row, col)
        else:
            self.current_position = self.initial_position

        self.current_velocity = (0, 0)
        return (self.current_position, self.current_velocity)

    def act(self, action):
        """
        Executes a single action (ax, ay).
        There is a 10% chance to override (ax, ay) with (0,0).
        Velocity is updated and clamped to [0, VELOCITY_LIMIT].
        Each step in row/column direction is checked for collisions or finish.
        If collision occurs, the car is reset. If finish is reached, the episode ends.
        Otherwise, returns (REWARD_MOVE, (position, velocity)).
        """
        # 10% chance to override action
        if np.random.rand() < 0.1:
            action = (0, 0)

        vx_new = max(min(self.current_velocity[0] + action[0], VELOCITY_LIMIT), 0)
        vy_new = max(min(self.current_velocity[1] + action[1], VELOCITY_LIMIT), 0)

        old_r, old_c = self.current_position
        row_step = -1 if vx_new > 0 else 0
        col_step = 1 if vy_new > 0 else 0

        steps_in_row = abs(vx_new)
        steps_in_col = abs(vy_new)

        r, c = old_r, old_c

        # Move in row direction
        for _ in range(steps_in_row):
            r += row_step
            result = self._check_cell(r, c)
            if result is not None:
                return result

        # Move in column direction
        for _ in range(steps_in_col):
            c += col_step
            result = self._check_cell(r, c)
            if result is not None:
                return result

        self.current_position = (r, c)
        self.current_velocity = (vx_new, vy_new)
        return (REWARD_MOVE, (self.current_position, self.current_velocity))

    def _check_cell(self, r, c):
        """
        Checks if (r, c) is out of bounds or a special cell:
        If out of bounds or 'E', returns (REWARD_OUT_OF_TRACK, reset state).
        If 'F', returns (REWARD_FINISH, None).
        Otherwise, returns None.
        """
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            new_start = self.reset()
            return (REWARD_OUT_OF_TRACK, new_start)

        cell_type = self.track_map[r, c]
        if cell_type == FINISH_LINE:
            return (REWARD_FINISH, None)
        elif cell_type == EDGE:
            new_start = self.reset()
            return (REWARD_OUT_OF_TRACK, new_start)

        return None

# -------------------- Helper Functions for Training -------------------- #

def action_gen():
    """
    Generates all possible actions (ax, ay) where ax, ay ∈ {-1, 0, 1}.
    """
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            yield (i, j)

def play(env, policy, max_steps=MAX_STEPS_PER_EPISODE):
    """
    Interacts with the environment until the episode ends or max_steps is reached.
    Returns (states, actions, rewards).
    """
    states = []
    actions = []
    rewards = []

    step_count = 0

    s0 = env.reset()
    a0 = policy(s0)

    states.append(s0)
    actions.append(a0)

    r_now, s_now = env.act(a0)
    step_count += 1

    while s_now is not None and step_count < max_steps:
        rewards.append(r_now)
        states.append(s_now)

        a_next = policy(s_now)
        actions.append(a_next)

        r_now, s_now = env.act(a_next)
        step_count += 1

    if s_now is None:
        rewards.append(r_now)
    else:
        rewards.append(REWARD_TIMEOUT)

    return states, actions, rewards

def build_avg_value_function(Q, track_map):
    """
    Builds an average value function map by averaging Q values over all actions.
    """
    rows, cols = track_map.shape
    V = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            avg_val = 0
            count = 0
            for vx in range(VELOCITY_LIMIT + 1):
                for vy in range(VELOCITY_LIMIT + 1):
                    for ax in [-1, 0, 1]:
                        for ay in [-1, 0, 1]:
                            key = (((i, j), (vx, vy)), (ax, ay))
                            if key in Q:
                                qv = Q[key]
                                count += 1
                                avg_val += (qv - avg_val) / count
            V[i, j] = avg_val
    return V

def build_max_value_function(Q, track_map):
    """
    Builds a max value function map by taking the maximum Q value over all actions.
    """
    rows, cols = track_map.shape
    V = np.zeros((rows, cols)) - 30
    for i in range(rows):
        for j in range(cols):
            candidates = [-30]
            for vx in range(VELOCITY_LIMIT + 1):
                for vy in range(VELOCITY_LIMIT + 1):
                    for ax in [-1, 0, 1]:
                        for ay in [-1, 0, 1]:
                            key = (((i, j), (vx, vy)), (ax, ay))
                            if key in Q:
                                candidates.append(Q[key])
            V[i, j] = max(candidates)
    return V

def plot_value_function(V, episode_count, tag):
    """
    Saves a heatmap of the value function to the date-based folder.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure()
    sns.heatmap(V, cmap="YlGnBu")
    plt.title(f"Value Function ({episode_count} episodes) - {tag}")
    file_name = os.path.join(RESULTS_FOLDER, f"value_function_{tag}_{episode_count}.png")
    plt.savefig(file_name)
    plt.close()

def mc_control(track_map, gamma=GAMMA, max_episode=EPISODES, env_name='env0'):
    """
    Off-policy Monte Carlo control with random behavior policy and importance sampling.
    """
    from collections import defaultdict

    Q = defaultdict(lambda: INITIAL_VALUE)
    C = defaultdict(float)

    pi = np.zeros((track_map.shape[0], track_map.shape[1], VELOCITY_LIMIT + 1, VELOCITY_LIMIT + 1), dtype=(int, 2))

    race_env = RaceTrack(track_map)

    for _ in tqdm(range(max_episode)):
        states, actions, rewards = play(
            race_env,
            policy=lambda s: (np.random.randint(-1, 2), np.random.randint(-1, 2)),
            max_steps=MAX_STEPS_PER_EPISODE
        )

        G = 0
        W = 1

        for st, at, rt in zip(reversed(states), reversed(actions), reversed(rewards)):
            G = gamma * G + rt
            stat = (st, at)

            C[stat] += W
            Q[stat] += (W / C[stat]) * (G - Q[stat])

            act_list = list(action_gen())
            q_values = [Q[(st, a)] for a in act_list]
            best_a = act_list[np.argmax(q_values)]
            pi[st[0][0], st[0][1], st[1][0], st[1][1]] = best_a

            if at != best_a:
                break

            W *= 9

    with open(env_name + '_policy.obj', 'wb') as f:
        pickle.dump(pi, f)

    # Plot average value
    V_avg = build_avg_value_function(Q, track_map)
    plot_value_function(V_avg, max_episode, tag=f"{env_name}_avg")

    # Plot max value
    V_max = build_max_value_function(Q, track_map)
    plot_value_function(V_max, max_episode, tag=f"{env_name}_max")

# -------------------- Trajectory Visualization -------------------- #

def act_with_substeps(race_env, action):
    """
    Executes action step by step to record intermediate positions.
    Returns (reward, new_state, sub_positions).
    """
    old_r, old_c = race_env.current_position
    old_vx, old_vy = race_env.current_velocity

    actual_action = action
    if np.random.rand() < 0.1:
        actual_action = (0, 0)

    vx_new = max(min(old_vx + actual_action[0], VELOCITY_LIMIT), 0)
    vy_new = max(min(old_vy + actual_action[1], VELOCITY_LIMIT), 0)

    sub_positions = []
    row_step = -1 if vx_new > 0 else 0
    col_step = 1 if vy_new > 0 else 0

    steps_in_row = abs(vx_new)
    steps_in_col = abs(vy_new)

    r, c = old_r, old_c

    for _ in range(steps_in_row):
        r += row_step
        result = race_env._check_cell(r, c)
        sub_positions.append((r, c))
        if result is not None:
            reward, s_new = result
            if s_new is None:
                return (reward, None, sub_positions)
            else:
                return (reward, s_new, sub_positions)

    for _ in range(steps_in_col):
        c += col_step
        result = race_env._check_cell(r, c)
        sub_positions.append((r, c))
        if result is not None:
            reward, s_new = result
            if s_new is None:
                return (reward, None, sub_positions)
            else:
                return (reward, s_new, sub_positions)

    race_env.current_position = (r, c)
    race_env.current_velocity = (vx_new, vy_new)
    return (REWARD_MOVE, (race_env.current_position, race_env.current_velocity), sub_positions)

def plot_trajectory_map(V, file_name):
    """
    Plots the trajectory on a heatmap and saves it to the date-based folder.
    """
    plt.figure(figsize=(V.shape[1] * 0.25, V.shape[0] * 0.25))
    numeric_map = np.zeros(V.shape, dtype=int)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            if V[i, j] == EDGE:
                numeric_map[i, j] = -2
            elif V[i, j] == TRACK_LINE:
                numeric_map[i, j] = 0
            elif V[i, j] == START_LINE:
                numeric_map[i, j] = -1
            elif V[i, j] == FINISH_LINE:
                numeric_map[i, j] = 10
            elif V[i, j] == 'X':
                numeric_map[i, j] = 5
            else:
                numeric_map[i, j] = 0

    sns.heatmap(numeric_map, cmap="YlGnBu", cbar=False, linewidths=0.1, linecolor='gray')
    plt.title('Path')
    full_path = os.path.join(RESULTS_FOLDER, file_name)
    plt.savefig(full_path)
    plt.close()

def generate_trajectory(track_map, policy_data, start_pos, env_name):
    """
    Generates a single trajectory from start_pos using the learned policy.
    Marks each sub-step on a copy of the track map.
    """
    from copy import deepcopy

    race_env = RaceTrack(track_map, start_pos)
    s0 = race_env.reset()
    V = deepcopy(track_map)

    V[s0[0][0], s0[0][1]] = 'X'

    (r, c), (vx, vy) = s0
    a0 = policy_data[r, c, vx, vy]
    rp, sp, subs = act_with_substeps(race_env, a0)

    for (rr, cc) in subs:
        V[rr, cc] = 'X'

    steps_count = 0
    while sp is not None and steps_count < MAX_TRAJECTORY_STEPS:
        (r, c), (vx, vy) = sp
        a_next = policy_data[r, c, vx, vy]
        rp, sp, subs = act_with_substeps(race_env, a_next)
        for (rr, cc) in subs:
            V[rr, cc] = 'X'
        steps_count += 1

    file_name = f"trajectory_{env_name}_startcol_{start_pos[1]}.png"
    plot_trajectory_map(V, file_name)

def generate_trajectory_plots(track_map, env_name):
    """
    Loads the saved policy and generates trajectory plots for each start cell on the last row.
    """
    with open(env_name + '_policy.obj', 'rb') as f:
        pi_data = pickle.load(f)

    last_row = track_map.shape[0] - 1
    for col in range(track_map.shape[1]):
        if track_map[last_row, col] == START_LINE:
            generate_trajectory(track_map, pi_data, (last_row, col), env_name)

# -------------------- Main Entry -------------------- #
if __name__ == '__main__':

    # TRAIN FIRST GRID
    grid1_str = """
    E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    0 0 0 0 0 0 0 0 0 0 E E E E E E E
    0 0 0 0 0 0 0 0 0 E E E E E E E E
    0 0 0 0 0 0 0 0 0 E E E E E E E E
    0 0 0 0 0 0 0 0 0 E E E E E E E E
    0 0 0 0 0 0 0 0 0 E E E E E E E E
    0 0 0 0 0 0 0 0 0 E E E E E E E E
    0 0 0 0 0 0 0 0 0 E E E E E E E E
    0 0 0 0 0 0 0 0 0 E E E E E E E E
    E 0 0 0 0 0 0 0 0 E E E E E E E E
    E 0 0 0 0 0 0 0 0 E E E E E E E E
    E 0 0 0 0 0 0 0 0 E E E E E E E E
    E 0 0 0 0 0 0 0 0 E E E E E E E E
    E 0 0 0 0 0 0 0 0 E E E E E E E E
    E 0 0 0 0 0 0 0 0 E E E E E E E E
    E 0 0 0 0 0 0 0 0 E E E E E E E E
    E 0 0 0 0 0 0 0 0 E E E E E E E E
    E E 0 0 0 0 0 0 0 E E E E E E E E
    E E 0 0 0 0 0 0 0 E E E E E E E E
    E E 0 0 0 0 0 0 0 E E E E E E E E
    E E 0 0 0 0 0 0 0 E E E E E E E E
    E E 0 0 0 0 0 0 0 E E E E E E E E
    E E 0 0 0 0 0 0 0 E E E E E E E E
    E E 0 0 0 0 0 0 0 E E E E E E E E
    E E E 0 0 0 0 0 0 E E E E E E E E
    E E E 0 0 0 0 0 0 E E E E E E E E
    E E E S S S S S S E E E E E E E E
    """

    track_map1 = np.array([[cell for cell in line.split()] for line in grid1_str.strip().splitlines()])
    mc_control(track_map1, gamma=1.0, max_episode=EPISODES, env_name='track_map1')
    generate_trajectory_plots(track_map1, env_name='track_map1')

    # TRAIN SECOND GRID
    grid2_str = """
    E E E E E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    E E E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    E E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    E E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    E E E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 F
    E E E E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 E E
    E E E E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E
    E E E E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E
    E E E E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 E E E E E E E E
    E E E E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 E E E E E E E E
    E E E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E
    E E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    E E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    E E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    E E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    E E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    E E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    E E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    E E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    E E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    E E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    E E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    E 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 E E E E E E E E E
    S S S S S S S S S S S S S S S S S S S S S S S E E E E E E E E E
    """

    track_map2 = np.array([[cell for cell in line.split()] for line in grid2_str.strip().splitlines()])
    mc_control(track_map2, gamma=1.0, max_episode=EPISODES, env_name='track_map2')
    generate_trajectory_plots(track_map2, env_name='track_map2')
