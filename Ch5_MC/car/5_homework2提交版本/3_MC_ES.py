import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from tqdm import trange

# Fix random seed for reproducibility
SEED = 42
np.random.seed(SEED)

STICK = 0
HIT = 1
ACTIONS = [STICK, HIT]

def draw_card():
    """
    Draw a card from an infinite deck (1..9 each 1/13, 10 accounts for 4/13).
    """
    c = np.random.randint(1, 14)  # 1..13
    return min(c, 10)

def hand_value(sum_without_ace, ace_count):
    """
    Given sum_without_ace (total of non-Ace cards) and ace_count (number of Aces, initially counted as 1 each),
    if there's an Ace and adding 10 won't cause a bust, mark it as usable and add 10 to the total.
    Returns (total_points, usable_ace)
    """
    total = sum_without_ace + ace_count
    usable = False
    if ace_count > 0 and total + 10 <= 21:
        total += 10
        usable = True
    return total, usable

def player_hit(player_sum, usable_ace):
    """
    Player takes a card. Returns (new_sum, new_usable_ace, bust).
    """
    card = draw_card()
    # Convert (player_sum, usable_ace) back to (sum_without_ace, ace_count)
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
    Dealer draws a hidden card first, then continues drawing until sum is >=17 or bust.
    Returns (dealer_sum, usable_ace).
    """
    second = draw_card()
    sum_without_ace = 0
    ace_count = 0

    # Face-up card
    if dealer_upcard == 1:
        ace_count += 1
    else:
        sum_without_ace += dealer_upcard

    # Hidden card
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
    Environment step:
    state = (player_sum, dealer_up, usable_ace)
    action = 0(STICK) or 1(HIT)
    Returns (next_state, reward, done)
    """
    player_sum, dealer_up, player_usable = state

    if action == HIT:
        new_sum, new_usable, bust = player_hit(player_sum, player_usable)
        if bust:
            return None, -1, True
        else:
            return (new_sum, dealer_up, new_usable), 0, False
    else:
        # STICK => dealer's turn
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
    Exploring Starts: Randomly select (s,a) from all [12..21]×[1..10]×{False,True} possibilities.
    """
    player_sum = np.random.randint(12, 22)
    dealer_up = np.random.randint(1, 11)
    ace = np.random.choice([False, True])
    action = np.random.choice(ACTIONS)
    return (player_sum, dealer_up, ace), action

def generate_episode_es_epsilon(Q, epsilon=0.05):
    """
    Generate an episode:
    1) Exploring Starts: initial (s,a) is random
    2) Then follow epsilon-greedy(Q) policy until termination
    Returns [(s0,a0,r1), (s1,a1,r2), ...]
    """
    episode = []
    state, action = random_state_action()
    done = False

    while True:
        next_state, reward, done = step(state, action)
        episode.append((state, action, reward))
        if done:
            break

        # Subsequent actions use epsilon-greedy(Q)
        state = next_state
        if np.random.rand() < epsilon:
            action = np.random.choice(ACTIONS)
        else:
            action = np.argmax(Q[state])

    return episode

def mc_es_epsilon_blackjack(num_episodes=5_000_000, epsilon=0.05):
    """
    MC ES + epsilon-greedy for subsequent decisions,
    default episode count is 5,000,000.
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

# ---------- 2×2 Visualization ----------
def plot_policy_value_2x2(policy, V):
    """
    Draw a 2×2 figure:
      Top row: usable ace
        Left: policy    Right: value function
      Bottom row: no usable ace
        Left: policy    Right: value function

    Policy plots: use color map to represent 0=Stick, 1=Hit
    Value plots: use gradient colors (e.g. 'viridis') to show magnitude of V(s)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Two rows (usable_ace=True / False), two columns (policy / value)
    for row_idx, ace in enumerate([True, False]):
        # --- First draw policy (col=0) ---
        ax_policy = axes[row_idx, 0]
        x_vals = range(1, 11)   # dealer up
        y_vals = range(12, 22)  # player sum
        policy_grid = np.zeros((len(y_vals), len(x_vals)), dtype=float)

        for i, d_up in enumerate(x_vals):
            for j, p_sum in enumerate(y_vals):
                s = (p_sum, d_up, ace)
                policy_grid[j, i] = policy.get(s, 0)

        # We can use a discrete cmap like 'Blues' with two color levels for 0 and 1
        # or 'binary', but for better contrast try 'coolwarm', 'cividis', etc.
        im1 = ax_policy.imshow(
            policy_grid,
            origin='lower',
            extent=[1, 10, 12, 21],
            cmap=plt.cm.Blues,
            vmin=0, vmax=1,  # Explicitly set range for 0 and 1
            aspect='auto'
        )
        ax_policy.set_xlabel("Dealer showing")
        ax_policy.set_ylabel("Player sum")
        ax_policy.set_title(f"Policy (usable_ace={ace})\n0=Stick, 1=Hit")

        # --- Then draw value function (col=1) ---
        ax_value = axes[row_idx, 1]
        value_grid = np.zeros((len(y_vals), len(x_vals)), dtype=float)
        for i, d_up in enumerate(x_vals):
            for j, p_sum in enumerate(y_vals):
                s = (p_sum, d_up, ace)
                value_grid[j, i] = V.get(s, 0.0)

        # Here we use 'viridis' to show value magnitude
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

        # Add colorbars
        fig.colorbar(im1, ax=ax_policy, fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax_value, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# ---------- Main Program ----------
if __name__ == "__main__":
    # 1) Train for 5,000,000 episodes with epsilon=0.05
    Q, policy = mc_es_epsilon_blackjack(num_episodes=5_000_000, epsilon=0.05)
    V = get_value_function(Q)

    # 2) Create 2×2 visualization
    plot_policy_value_2x2(policy, V)