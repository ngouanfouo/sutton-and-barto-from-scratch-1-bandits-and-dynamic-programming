"""
Sutton and Barto from Scratch 1: Bandits and Dynamic Programming

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - create_bandit_testbed
import numpy as np

def create_bandit_testbed(k: int, seed: int, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """
    Creates a k-armed bandit testbed by sampling true action values from a 
    normal distribution using the legacy RandomState API for exact sequence matching.
    
    Args:
        k: The number of actions/arms in the multi-armed bandit.
        seed: Random seed for reproducibility.
        mean: Mean of the normal distribution (default 0.0).
        std: Standard deviation of the normal distribution (default 1.0).
        
    Returns:
        A 1-D NumPy array of shape (k,) containing the true action values.
    """
    # Use the legacy RandomState to exactly replicate the expected testbed sequence
    rng = np.random.RandomState(seed)
    
    # Draw k samples using the legacy normal generator
    true_values = rng.normal(loc=mean, scale=std, size=k)
    
    return true_values

# Step 2 - pull_arm
import numpy as np

def pull_arm(true_values: np.ndarray, action: int, rng: np.random.Generator) -> float:
    """Pull one arm and return reward = true value + unit-normal noise.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        action (int): Index of the arm to pull.
        rng (np.random.Generator): Seeded random generator for the noise.

    Returns:
        float: Stochastic reward for this pull.
    """
    # Sample from a standard normal distribution (mean=0.0, std=1.0)
    noise = rng.normal()
    
    # Return the true value of the selected arm plus the stochastic noise
    return float(true_values[action] + noise)

# Step 3 - sample_average_update
import numpy as np

def sample_average_update(q_values: np.ndarray, action_counts: np.ndarray, action: int, reward: float) -> tuple:
    """Update an action-value estimate incrementally from one new reward.

    Args:
        q_values (np.ndarray): Shape (k,) current action-value estimates.
        action_counts (np.ndarray): Shape (k,) number of times each arm has been chosen.
        action (int): Index of the arm that was selected.
        reward (float): Observed reward scalar.

    Returns:
        tuple: (updated_q_values, updated_action_counts) as clean, independent copies.
    """
    # Create copies to prevent unexpected shared mutating side effects
    updated_q_values = q_values.copy().astype(float)
    updated_action_counts = action_counts.copy().astype(int)
    
    # Increment the visitation counter for the selected arm
    updated_action_counts[action] += 1
    
    # Compute the incremental step size: alpha_n = 1 / N(a)
    n = updated_action_counts[action]
    
    # Q_{n+1} = Q_n + (1 / n) * (R_n - Q_n)
    updated_q_values[action] += (reward - updated_q_values[action]) / n
    
    return updated_q_values, updated_action_counts

# Step 4 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:

    k = len(q_values)
    
    # Decide between exploration or exploitation
    if rng.random() < epsilon:
        # Exploration: choose a uniformly random action among all k arms
        return int(rng.integers(0, k))
    else:
        # Exploitation: select the greedy action (breaking ties by picking the smallest index)
        # np.argmax inherently returns the first (smallest) index matching the maximum value
        return int(np.argmax(q_values))

# Step 5 - run_bandit_episode
import numpy as np

def run_bandit_episode(true_values: np.ndarray, n_steps: int, epsilon: float, rng: np.random.Generator) -> tuple:
    """Run one bandit episode with epsilon-greedy selection and sample-average updates.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, actions) with shapes (n_steps,) and (n_steps,) of ints.
    """
    k = len(true_values)
    
    # Initialize tracking structures
    q_values = np.zeros(k, dtype=float)
    action_counts = np.zeros(k, dtype=int)
    
    # Pre-allocate output buffers
    rewards_history = np.zeros(n_steps, dtype=float)
    actions_history = np.zeros(n_steps, dtype=int)
    
    for t in range(n_steps):
        # 1. Select arm epsilon-greedily based on current value estimates
        action = epsilon_greedy_action(q_values, epsilon, rng)
        
        # 2. Execute pull to obtain stochastic reward
        reward = pull_arm(true_values, action, rng)
        
        # 3. Apply sample-average incremental updates to value tracking matrices
        q_values, action_counts = sample_average_update(q_values, action_counts, action, reward)
        
        # 4. Store current interaction records
        actions_history[t] = action
        rewards_history[t] = reward
        
    return rewards_history, actions_history

# Step 6 - track_rewards_and_optimal_actions
import numpy as np

def track_rewards_and_optimal_actions(true_values: np.ndarray, n_steps: int, epsilon: float, rng: np.random.Generator) -> tuple:
    """Run one episode tracking rewards and optimal-arm choices.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, optimal_flags) each shape (n_steps,).
            optimal_flags entries are 0.0 or 1.0 floats.
    """
    # 1. Identify the optimal arm index (breaking ties using the smallest index, consistent with argmax)
    optimal_action = np.argmax(true_values)
    
    # 2. Collect interaction history from the running environment
    rewards, actions = run_bandit_episode(true_values, n_steps, epsilon, rng)
    
    # 3. Create a 0/1 indicator mask indicating whether the chosen action matches the optimal action
    optimal_flags = (actions == optimal_action).astype(float)
    
    return rewards, optimal_flags

# Step 7 - average_bandit_curves
import numpy as np

def average_bandit_curves(k: int, n_runs: int, n_steps: int, epsilon: float, seed: int) -> tuple:
    """Average reward and optimal-action curves over many independent bandit runs.

    Args:
        k (int): Number of arms in the bandit environment.
        n_runs (int): Number of independent simulation runs to average over.
        n_steps (int): Number of time steps per run horizon.
        epsilon (float): Exploration probability for epsilon-greedy selection.
        seed (int): Base random seed value used for reproducibility.

    Returns:
        tuple: (mean_rewards, mean_optimal_fractions) each shape (n_steps,).
    """
    # Pre-allocate matrices to accumulate time-step vectors over all runs
    all_rewards = np.zeros((n_runs, n_steps), dtype=float)
    all_optimal_flags = np.zeros((n_runs, n_steps), dtype=float)
    
    for i in range(n_runs):
        # Generate independent seeded components unique to this specific run
        run_seed = seed + i
        
        # 1. Initialize the k-armed testbed distribution values
        true_values = create_bandit_testbed(k, run_seed)
        
        # 2. Spawn an independent stochastic number generator for the episode
        rng = np.random.default_rng(run_seed)
        
        # 3. Simulate the episode interaction trajectory
        rewards, optimal_flags = track_rewards_and_optimal_actions(
            true_values, n_steps, epsilon, rng
        )
        
        # Store tracking histories
        all_rewards[i] = rewards
        all_optimal_flags[i] = optimal_flags
        
    # Aggregate data across the runs dimension (axis 0) to compute temporal averages
    mean_rewards = np.mean(all_rewards, axis=0)
    mean_optimal_fractions = np.mean(all_optimal_flags, axis=0)
    
    return mean_rewards, mean_optimal_fractions

# Step 8 - apply_random_walk_drift
import numpy as np

def apply_random_walk_drift(true_values: np.ndarray, drift_std: float, rng: np.random.Generator) -> np.ndarray:

    # Sample zero-mean normal noise matching the shape of the true action values
    noise = rng.normal(loc=0.0, scale=drift_std, size=true_values.shape)
    
    # Return a new array with the independent random-walk step incorporated
    return true_values + noise

# Step 9 - constant_step_size_update
import numpy as np

def constant_step_size_update(q_values: np.ndarray, action: int, reward: float, alpha: float) -> np.ndarray:
    """Apply the constant step-size update to the selected action.

    Args:
        q_values (np.ndarray): Shape (k,) current action-value estimates.
        action (int): Index of the arm that was selected.
        reward (float): Observed reward scalar.
        alpha (float): Constant step-size parameter (learning rate) in (0, 1].

    Returns:
        np.ndarray: A new array of shape (k,) with the updated action-value estimates.
    """
    # Create a copy to prevent unintended side effects from mutating the original array
    updated_q_values = q_values.copy().astype(float)
    
    # Q_{n+1} = Q_n + alpha * (R_n - Q_n)
    updated_q_values[action] += alpha * (reward - updated_q_values[action])
    
    return updated_q_values

# Step 10 - optimistic_initialization (not yet solved)
# TODO: implement

# Step 11 - ucb_action_select (not yet solved)
# TODO: implement

# Step 12 - gradient_bandit_update (not yet solved)
# TODO: implement

# Step 13 - bandit_parameter_study (not yet solved)
# TODO: implement

# Step 14 - build_gridworld_mdp (not yet solved)
# TODO: implement

# Step 15 - iterative_policy_evaluation (not yet solved)
# TODO: implement

# Step 16 - greedy_policy_improvement (not yet solved)
# TODO: implement

# Step 17 - policy_iteration (not yet solved)
# TODO: implement

# Step 18 - value_iteration (not yet solved)
# TODO: implement

# Step 19 - build_gambler_mdp (not yet solved)
# TODO: implement

# Step 20 - gambler_value_iteration (not yet solved)
# TODO: implement

# Step 21 - extract_optimal_stakes (not yet solved)
# TODO: implement

