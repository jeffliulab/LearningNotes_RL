# Homework3 Codes and Files

## Requirements

The scripts require **Python** with the following libraries:
- `numpy` for numerical computations
- `matplotlib` for plotting results
- `seaborn` for heatmaps
- `pickle` for saving/loading policies
- `tqdm` for progress tracking (optional, can be removed if not available)

## Files and Usage

### Files

- **RL_HW2_PangLiu.pdf**: The paper in latex format.
- **50000_episodes_test_results**: All outputs of the `1_racetrack.py`

### 1. **`1_racetrack.py`**
   - Implements the **Racetrack** environment as a **grid-based simulation**.
   - Uses **Off-Policy Monte Carlo Control** with importance sampling to learn optimal policies.
   - Defines a **RaceTrack class** that manages car movement, velocity updates, and collisions.
   - Generates **value function heatmaps** and **racecar trajectory visualizations**.
   
   **Main functions:**
   - `mc_control()`: Runs the Monte Carlo learning process.
   - `play()`: Simulates episodes following the learned policy.
   - `generate_trajectory_plots()`: Visualizes learned racecar trajectories.
   
   **Usage:**
   ```bash
   python 1_racetrack.py
   ```
   Runs the training process on predefined racetrack environments and generates heatmaps and trajectory plots.

### 2. **`2_blackjack.py`**
   - Implements **Monte Carlo policy evaluation** for **Blackjack**.
   - Simulates multiple Blackjack hands and learns the value function.
   - Uses a **first-visit Monte Carlo** approach.
   - Generates **3D state-value function visualizations**.
   
   **Main functions:**
   - `simulate_episode()`: Simulates a Blackjack game and records state transitions.
   - `monte_carlo_blackjack()`: Estimates state-value functions from simulated episodes.
   - `plot_value_function()`: Generates 3D surface plots of the learned value function.
   
   **Usage:**
   ```bash
   python 2_blackjack.py
   ```
   Runs the Blackjack Monte Carlo simulation and generates visualizations.

### 3. **`3_MC_ES.py`**
   - Implements **Monte Carlo Exploring Starts (MC-ES) for Blackjack**.
   - Uses an **off-policy approach with ε-greedy exploration**.
   - Trains a policy using **Exploring Starts** to ensure all states are visited.
   - Produces **policy heatmaps** and **value function visualizations**.
   
   **Main functions:**
   - `mc_es_epsilon_blackjack()`: Runs MC-ES learning with an epsilon-greedy approach.
   - `get_value_function()`: Computes state-value estimates from learned action-values.
   - `plot_policy_value_2x2()`: Visualizes learned policies and value functions.
   
   **Usage:**
   ```bash
   python 3_MC_ES.py
   ```
   Runs the MC-ES learning process for Blackjack and generates policy/value visualizations.

## How to Use

Each script runs an independent reinforcement learning experiment and generates corresponding plots. 

### Outputs include:
- **Racetrack:** Value function heatmaps and trajectory visualizations.
- **Blackjack:** 3D state-value function plots.
- **MC-ES Blackjack:** Policy and value function visualizations.

The scripts can be executed directly in a Python environment or in Jupyter Notebook for better visualization.

## Notes
- The experiments use **ε-greedy action selection (ε=0.05 for MC-ES Blackjack)**.
- Each experiment runs for **50,000 episodes** for policy evaluation and control.
- `tqdm` is used for a progress bar; if not installed, the corresponding function calls can be removed.