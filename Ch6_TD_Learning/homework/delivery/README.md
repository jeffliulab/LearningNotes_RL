
# Homework3 Codes and Files

## Requirements

The scripts require **Python** with the following libraries:
- `numpy` for numerical computations
- `matplotlib` for plotting results
- `tqdm` for progress tracking (optional; can be removed if not available)
- `pygame` (required for **2_secondary.py**) for the graphical demo

## Files and Usage

### Files

- **README.md**: This file.
- **1_primary.py**: Implements the grid world environment with wind simulation and reinforcement learning.
- **2_secondary.py**: Implements a discrete version of the Breakout (brick breaking) game using Q-learning.

### 1. `1_primary.py`
- **Description:**
  - Implements a **grid world environment** (10x10) with a stochastic wind that affects the agent's movement.
  - Combines ideas from King's Moves (allowing 8-directional moves plus a "stay" action) with stochastic wind (each column has a base wind strength that may randomly vary).
  - Uses three reinforcement learning methods: **SARSA**, **Q-learning**, and **Monte Carlo**.
  - Contains functions such as:
    - `run_experiment()`: Runs the training process over multiple episodes and runs.
    - Q-value update routines for SARSA, Q-learning, and Monte Carlo.
    - Plotting routines that generate learning curves (average reward and success rate vs. episodes).
- **Usage:**
  ```bash
  python 1_primary.py
  ```
  Running this script will train the agent on the predefined grid world and generate plots that compare the performance of the three RL methods.

### 2. `2_secondary.py`
- **Description:**
  - Implements a discrete version of the **Breakout game** on a 10x10 grid.
  - The environment includes:
    - A paddle fixed at the bottom row that can move left or right.
    - Bricks arranged in the top row (represented by a bit mask).
    - A ball that moves with a fixed velocity and bounces off walls, bricks, and the paddle.
  - Uses a **Q-learning algorithm** with an ε-greedy strategy (with decaying ε) to train an agent that controls the paddle.
  - Provides a graphical demo using **pygame** to visualize the agent playing the game.
  - Also plots learning curves (episode reward and success rate) over the training episodes.
- **Usage:**
  ```bash
  python 2_secondary.py
  ```
  Running this script will:
  - Train the Q-learning agent over a large number of episodes (e.g., 100,000 episodes).
  - Display a demo of the game with the learned policy using pygame.
  - Generate and display learning curves that show the agent’s performance during training.
