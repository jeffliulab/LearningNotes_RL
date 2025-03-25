"""
注意, 这个只是对一个确定的policy的Monte Carlo评估, 即评估该策略的价值函数。
"""

import random
import matplotlib.pyplot as plt
import numpy as np

# ----------------- Blackjack Simulation Functions -----------------

def draw_card():
    """Draw a card from an infinite deck: numbers 1-10, where 10 represents all face cards."""
    card = random.randint(1, 13)
    return card if card < 10 else 10

def draw_hand():
    """Draw two cards to form an initial hand."""
    return [draw_card(), draw_card()]

def usable_ace(hand):
    """Return True if the hand has an ace counted as 11 (i.e., using the ace as 11 doesn’t bust)."""
    return 1 in hand and sum(hand) + 10 <= 21

def sum_hand(hand):
    """Calculate the sum of the hand; count one ace as 11 if it is usable."""
    s = sum(hand)
    return s + 10 if usable_ace(hand) else s

def is_bust(hand):
    """Return True if the hand is over 21."""
    return sum_hand(hand) > 21

def dealer_policy(dealer_hand):
    """Dealer hits until reaching 17 or more."""
    while sum_hand(dealer_hand) < 17:
        dealer_hand.append(draw_card())
    return dealer_hand

def simulate_episode():
    """
    Simulate one episode of Blackjack.
    
    Returns:
      states: a list of states encountered by the player. Each state is a tuple:
              (player sum, dealer's showing card, usable ace boolean)
      reward: the final reward (+1 for win, -1 for loss, 0 for draw)
    """
    # Initialize player and dealer hands
    player_hand = draw_hand()
    dealer_hand = draw_hand()
    
    # Record states encountered during the player's turn.
    states = []
    state = (sum_hand(player_hand), dealer_hand[0], usable_ace(player_hand))
    states.append(state)
    
    # Player's turn: hit if sum <= 19, stick otherwise.
    while sum_hand(player_hand) <= 19:
        player_hand.append(draw_card())
        if is_bust(player_hand):
            return states, -1  # Player busts: lose immediately.
        state = (sum_hand(player_hand), dealer_hand[0], usable_ace(player_hand))
        states.append(state)
    
    # Player sticks: now it's the dealer's turn.
    dealer_hand = dealer_policy(dealer_hand)
    if is_bust(dealer_hand):
        reward = 1  # Dealer busts: player wins.
    else:
        player_score = sum_hand(player_hand)
        dealer_score = sum_hand(dealer_hand)
        if player_score > dealer_score:
            reward = 1
        elif player_score < dealer_score:
            reward = -1
        else:
            reward = 0
    return states, reward

def monte_carlo_blackjack(episodes):
    """
    Run the Monte Carlo simulation for a given number of episodes.
    
    Returns:
      V: dictionary of estimated state values V(s) as the average of returns.
      Returns: dictionary where each key is a state and the value is a list of returns.
    """
    Returns = {}  # Dictionary to store lists of returns for each state.
    V = {}        # Dictionary to store value estimates for each state.
    
    for episode in range(episodes):
        states, reward = simulate_episode()
        # First-visit Monte Carlo: update only on the first occurrence of each state.
        visited = set()
        for s in states:
            if s not in visited:
                Returns.setdefault(s, []).append(reward)
                visited.add(s)
    
    # Compute V(s) as the average return for each state.
    for s, rewards in Returns.items():
        V[s] = sum(rewards) / len(rewards)
    return V, Returns

# ----------------- Visualization Functions -----------------

def plot_value_function(V, usable_ace_flag):
    """
    Prepare grid data for the state-value function for a given usable ace flag.
    
    Parameters:
      V: dictionary with state values
      usable_ace_flag: Boolean, True for states with a usable ace, False otherwise
      
    Returns:
      X, Y, Z: meshgrid arrays for plotting, where
               X: dealer's showing card (1-10),
               Y: player's sum (typically 12-21),
               Z: estimated state value V(s)
    """
    # Define the grid ranges for player's sum and dealer's showing card.
    player_sums = np.arange(12, 22)  # typically player's sum between 12 and 21
    dealer_cards = np.arange(1, 11)    # dealer's showing card between 1 and 10
    
    X, Y = np.meshgrid(dealer_cards, player_sums)
    Z = np.zeros_like(X, dtype=float)
    
    for i, player_sum in enumerate(player_sums):
        for j, dealer in enumerate(dealer_cards):
            state = (player_sum, dealer, usable_ace_flag)
            Z[i, j] = V.get(state, 0)
    return X, Y, Z

# ----------------- Main Training and Plotting -----------------

if __name__ == '__main__':
    episodes = 50000
    print("Training for", episodes, "episodes...")
    V, Returns = monte_carlo_blackjack(episodes)
    print("Training completed. Preparing visualization...")

    # Create a 3D plot for states with a usable ace and without a usable ace.
    fig = plt.figure(figsize=(14, 6))

    # Plot for states with a usable ace.
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y, Z = plot_value_function(V, True)
    surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_title("State-Value Function (Usable Ace = True)")
    ax1.set_xlabel("Dealer Showing")
    ax1.set_ylabel("Player Sum")
    ax1.set_zlabel("Estimated Value")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Plot for states without a usable ace.
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y, Z = plot_value_function(V, False)
    surf2 = ax2.plot_surface(X, Y, Z, cmap='viridis')
    ax2.set_title("State-Value Function (Usable Ace = False)")
    ax2.set_xlabel("Dealer Showing")
    ax2.set_ylabel("Player Sum")
    ax2.set_zlabel("Estimated Value")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()
