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

# Step 10 - optimistic_initialization
import numpy as np

def optimistic_initialization(k: int, initial_value: float) -> np.ndarray:
    """Create starting action-value estimates that drive early exploration.

    Args:
        k (int): Number of arms in the bandit environment.
        initial_value (float): The high, optimistic initial value for all estimates.

    Returns:
        np.ndarray: A 1-D array of shape (k,) filled with initial_value.
    """
    # Create a pre-allocated array of shape (k,) initialized with the given starting value
    return np.full(shape=(k,), fill_value=float(initial_value), dtype=float)

# Step 11 - ucb_action_select
import numpy as np

def ucb_action_select(q_values: np.ndarray, action_counts: np.ndarray, timestep: int, c: float) -> int:
    """Select an action by upper-confidence-bound scores.

    Args:
        q_values (np.ndarray): Action-value estimates, shape (k,).
        action_counts (np.ndarray): Visit counts per action, shape (k,).
        timestep (int): Current time step t (>= 1).
        c (float): Exploration constant.

    Returns:
        int: Index of the selected action.
    """
    k = len(q_values)
    
    # 1. Identify unvisited arms where N(a) == 0
    unvisited_mask = (action_counts == 0)
    
    if np.any(unvisited_mask):
        # Tie-break rule: Pick the smallest index among the unvisited arms
        return int(np.argmax(unvisited_mask))
        
    # 2. Compute the UCB score for all arms since all N(a) > 0
    # Score = Q(a) + c * sqrt(ln(t) / N(a))
    ln_t = np.log(timestep)
    variance_bounds = np.sqrt(ln_t / action_counts)
    ucb_scores = q_values + c * variance_bounds
    
    # 3. Select the arm maximizing the score (np.argmax breaks ties by returning the first/smallest index)
    return int(np.argmax(ucb_scores))

# Step 12 - gradient_bandit_update
import numpy as np

def gradient_bandit_update(
    preferences: np.ndarray, 
    action: int, 
    reward: float, 
    average_reward: float, 
    alpha: float
) -> np.ndarray:
    """Update softmax action preferences with one gradient-bandit step.

    Args:
        preferences (np.ndarray): Shape (k,) current action preferences H_t(a).
        action (int): Index of the arm that was selected.
        reward (float): Observed reward scalar R_t.
        average_reward (float): Baseline average reward over time (scalar).
        alpha (float): Step-size parameter (learning rate) > 0.

    Returns:
        np.ndarray: Updated preference vector of shape (k,).
    """
    # 1. Compute numerically stable softmax probabilities to get the policy
    # Subtracting the maximum preference prevents floating-point overflow
    shifted_prefs = preferences - np.max(preferences)
    exp_prefs = np.exp(shifted_prefs)
    policy = exp_prefs / np.sum(exp_prefs)
    
    # 2. Compute the baseline advantage signal
    advantage = reward - average_reward
    
    # 3. Apply the gradient update step
    # H_{t+1}(a) = H_t(a) - alpha * (R_t - baseline) * pi_t(a)  for all a != A_t
    # H_{t+1}(A_t) = H_t(A_t) + alpha * (R_t - baseline) * (1 - pi_t(A_t))
    updated_preferences = preferences.copy().astype(float)
    
    # Apply the general penalty/decay to all options based on their selection odds
    updated_preferences -= alpha * advantage * policy
    
    # Offset the chosen action to match the targeted update indicator rule
    updated_preferences[action] += alpha * advantage
    
    return updated_preferences

# Step 13 - bandit_parameter_study
import numpy as np
from typing import List, Dict, Any

def bandit_parameter_study(n_runs: int, n_steps: int, seed: int, settings: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compare multiple bandit strategies over independent runs and return final-step mean rewards.

    Args:
        n_runs (int): Number of independent runs to average over.
        n_steps (int): Total number of pulling steps in each episode.
        seed (int): Base random seed for reproducibility.
        settings (List[Dict[str, Any]]): List of dicts specifying 'method', 'param', 
                                         and optional 'nonstationary' (bool).

    Returns:
        Dict[str, float]: Maps configuration string label to its final step mean reward.
    """
    k = 10  # Standard 10-armed testbed setup
    results = {}
    
    # Pre-generate true values across all runs to isolate policy differences from testbed variance
    # Generate an explicit array of random seeds for each run to cleanly fork child streams
    master_rng = np.random.default_rng(seed)
    run_seeds = master_rng.integers(0, 2**32 - 1, size=n_runs)
    
    # Initialize baseline stationary true values for all independent testbeds upfront
    base_true_values = np.zeros((n_runs, k))
    for r_idx in range(n_runs):
        run_rng = np.random.default_rng(run_seeds[r_idx])
        base_true_values[r_idx] = run_rng.normal(0.0, 1.0, size=k)

    for setting in settings:
        method = setting['method']
        param = setting['param']
        nonstationary = setting.get('nonstationary', False)
        
        label = f"{method}({param})"
        if nonstationary:
            label += ",ns"
            
        run_rewards = np.zeros((n_runs, n_steps))
        
        for run in range(n_runs):
            # Fork a distinct child RNG stream for action selection and reward noise
            agent_rng = np.random.default_rng(run_seeds[run])
            
            # Setup initial arms values
            if nonstationary:
                true_values = np.zeros(k, dtype=float)
            else:
                true_values = base_true_values[run].copy()
                
            # Initialize agent value/preference matrices
            q_values = np.zeros(k, dtype=float)
            action_counts = np.zeros(k, dtype=int)
            preferences = np.zeros(k, dtype=float)
            running_average_reward = 0.0
            
            if method == 'optimistic':
                q_values = np.full(k, fill_value=float(param), dtype=float)
                
            for step in range(1, n_steps + 1):
                # Drift environment values if working under non-stationary regimes
                if nonstationary and step > 1:
                    true_values += agent_rng.normal(0.0, 0.01, size=k)
                    
                # --- Policy Action Selection ---
                if method == 'epsilon_greedy':
                    if agent_rng.random() < param:
                        action = agent_rng.integers(0, k)
                    else:
                        action = int(np.argmax(q_values))
                        
                elif method == 'constant_step':
                    if agent_rng.random() < 0.1:  # Fixed baseline exploration
                        action = agent_rng.integers(0, k)
                    else:
                        action = int(np.argmax(q_values))
                        
                elif method == 'optimistic':
                    action = int(np.argmax(q_values))
                    
                elif method == 'ucb':
                    unvisited = (action_counts == 0)
                    if np.any(unvisited):
                        action = int(np.argmax(unvisited))
                    else:
                        ucb_scores = q_values + param * np.sqrt(np.log(step) / action_counts)
                        action = int(np.argmax(ucb_scores))
                        
                elif method == 'gradient':
                    shifted = preferences - np.max(preferences)
                    exp_p = np.exp(shifted)
                    policy = exp_p / np.sum(exp_p)
                    action = agent_rng.choice(k, p=policy)
                
                # --- Step Performance & Feedback ---
                reward = agent_rng.normal(true_values[action], 1.0)
                run_rewards[run, step - 1] = reward
                
                # --- Value Matrix / Preference Updates ---
                action_counts[action] += 1
                
                if method == 'epsilon_greedy' or method == 'ucb':
                    alpha_n = 1.0 / action_counts[action]
                    q_values[action] += alpha_n * (reward - q_values[action])
                    
                elif method == 'constant_step' or method == 'optimistic':
                    alpha_step = param if method == 'constant_step' else 0.1
                    q_values[action] += alpha_step * (reward - q_values[action])
                    
                elif method == 'gradient':
                    running_average_reward += (reward - running_average_reward) / step
                    adv = reward - running_average_reward
                    preferences -= param * adv * policy
                    preferences[action] += param * adv
                    
        results[label] = float(np.mean(run_rewards[:, -1]))
        
    return results

# Step 14 - build_gridworld_mdp
from typing import Dict, List, Tuple, Any

def build_gridworld_mdp() -> Dict[str, Any]:
    """Construct the classic 4x4 Sutton & Barto gridworld as an MDP dynamics table.
    
    States are 0..15 in row-major order. States 0 and 15 are terminal sinks.
    Actions: 0=North, 1=East, 2=South, 3=West.
    
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'n_states': int (16)
            - 'n_actions': int (4)
            - 'P': Dict[int, Dict[int, List[Tuple[float, int, float]]]]
                   Mapping s -> a -> [(probability, next_state, reward)]
    """
    n_states = 16
    n_actions = 4
    
    # Define action movements as coordinate updates: (delta_row, delta_col)
    # 0 = North (row - 1), 1 = East (col + 1), 2 = South (row + 1), 3 = West (col - 1)
    action_moves = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1)
    }
    
    P = {}
    
    for s in range(n_states):
        P[s] = {}
        
        for a in range(n_actions):
            # 1. Handle absorbing terminal states (0 and 15)
            if s in (0, 15):
                P[s][a] = [(1.0, s, 0.0)]
                continue
            
            # 2. Map 1D state back to 2D grid dimensions (4x4)
            r, c = s // 4, s % 4
            dr, dc = action_moves[a]
            
            # Compute tentative target position
            next_r = r + dr
            next_c = c + dc
            
            # 3. Apply bounding walls check
            if 0 <= next_r < 4 and 0 <= next_c < 4:
                next_state = next_r * 4 + next_c
            else:
                next_state = s  # Bounce off the wall and remain in place
                
            # Every active transaction inside the gridworld incurs a step penalty of -1.0
            P[s][a] = [(1.0, next_state, -1.0)]
            
    return {
        "n_states": n_states,
        "n_actions": n_actions,
        "P": P
    }

# Step 15 - iterative_policy_evaluation
import numpy as np
from typing import Dict, Any

def iterative_policy_evaluation(
    policy: np.ndarray, 
    mdp: Dict[str, Any], 
    gamma: float, 
    theta: float
) -> np.ndarray:
    """Compute the state-value function of a fixed policy using synchronous or 
    in-place successive sweeps until maximum change falls below theta.

    Args:
        policy (np.ndarray): Shape (n_states,) containing deterministic action indices,
                             OR shape (n_states, n_actions) containing action probabilities.
        mdp (Dict[str, Any]): MDP table matching the structure of `build_gridworld_mdp`.
        gamma (float): Discount factor in [0, 1].
        theta (float): Convergence threshold factor > 0.

    Returns:
        np.ndarray: Converged state-value function vector of shape (n_states,).
    """
    n_states = mdp['n_states']
    n_actions = mdp['n_actions']
    P = mdp['P']
    
    # Initialize the value state vector with zeros
    V = np.zeros(n_states, dtype=float)
    
    # Determine policy representation structure
    is_deterministic = (policy.ndim == 1)
    
    while True:
        delta = 0.0
        
        for s in range(n_states):
            v_old = V[s]
            v_new = 0.0
            
            if is_deterministic:
                # Deterministic Policy: action is fixed for state s
                a = int(policy[s])
                # Expectation over transitions: sum_{s', r} p(s', r | s, a) * [r + gamma * V(s')]
                for prob, next_state, reward in P[s][a]:
                    v_new += prob * (reward + gamma * V[next_state])
            else:
                # Stochastic Policy: sum over actions and transitions
                for a in range(n_actions):
                    action_prob = policy[s, a]
                    if action_prob > 0.0:
                        for prob, next_state, reward in P[s][a]:
                            v_new += action_prob * prob * (reward + gamma * V[next_state])
            
            V[s] = v_new
            delta = max(delta, abs(v_old - v_new))
            
        # Terminate when the maximum value change over a full sweep is strictly less than theta
        if delta < theta:
            break
            
    return V

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

