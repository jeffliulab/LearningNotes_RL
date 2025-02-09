# Reinforcement Learning: Multi-Armed Bandit Experiments

The five scripts contains implementations of different Q-value estimation methods for solving the multi-armed bandit problem in a **non-stationary environment**. The experiments compare various approaches, including sample averaging, fixed step-size updates, adaptive step-size, and sliding window methods.

## Requirements

The scripts require **Python** with the following libraries:  
- `numpy` for numerical computations  
- `matplotlib` for plotting results  
- `tqdm` for progress tracking (optional, can be removed if not available)  

## Files and Usage

1. **`1_experiment_2_2.py`** 
   - Implements a 10-armed bandit problem with a stationary reward distribution.  
   - Compares **sample averaging** and **fixed step-size (α=0.1)** methods.  
   - Generates reward and optimal action selection plots.

2. **`2_excersize_2_5.py`** (Excersize 2.5)  
   - Extends the baseline to a **non-stationary environment** where Q-values drift over time.  
   - Compares **sample averaging, fixed step-size, and adaptive step-size (α_t = 1/t)**.  
   - Includes smoothed plots for better visualization.

3. **`3_adaptive.py`**  
   - Introduces a **sliding window approach** to handle non-stationary problems.  
   - Compares sample averaging, fixed step-size, adaptive step-size, and sliding window methods.

4. **`4_sliding_window.py`**  
   - Runs experiments for **different values of fixed step-size α** to find the optimal learning rate.  
   - Tests α = {0.01, 0.05, 0.1, 0.2, 0.5} and generates performance plots.

5. **`5_alpha.py`**  
   - Generates **smoothed performance plots** for all tested methods.  
   - Uses moving average to reduce noise and improve trend visualization.

## How to Use

Each script runs an experiment and generates corresponding plots.  
The main outputs include:  
- **Average reward over steps**  
- **Optimal action selection percentage over steps**  

The scripts are standalone and can be executed directly in a Python environment.  

You can also use Jupyter Notebook to run, in that way you can store the figure images and modify plot or training part easily.

## Notes

- The experiments use **ε-greedy action selection (ε=0.1)**.  
- Each experiment runs for **10,000 steps** and averages over **2,000 runs** for reliable results.  
- `tqdm` is used for a progress bar; if not installed, the corresponding `tqdm` function can be removed.  