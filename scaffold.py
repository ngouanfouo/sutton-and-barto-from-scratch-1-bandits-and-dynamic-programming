"""
Sutton and Barto from Scratch 1: Bandits and Dynamic Programming scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Sutton & Barto from scratch: multi-armed bandits and DP demo."""
import numpy as np


def main():
    np.random.seed(0)

    # --- Stationary k-armed bandit ---
    k = 10
    true_values = create_bandit_testbed(k, seed=0)
    print("True action values:", np.round(true_values, 3))

    rng = np.random.default_rng(1)
    rewards = run_bandit_episode(true_values, n_steps=200, epsilon=0.1, rng=rng)
    print("Episode mean reward (eps=0.1):", round(float(np.mean(rewards)), 4))

    avg_r, avg_opt = average_bandit_curves(
        k=10, n_runs=50, n_steps=200, epsilon=0.1, seed=0
    )
    print("Avg reward @200:", round(float(avg_r[-1]), 4))
    print("Optimal action % @200:", round(float(avg_opt[-1]), 4))

    # Nonstationary step + constant-step-size / optimistic / UCB / gradient pieces
    drift_rng = np.random.default_rng(2)
    drifted = apply_random_walk_drift(true_values.copy(), drift_std=0.01, rng=drift_rng)
    print("Mean |drift|:", round(float(np.mean(np.abs(drifted - true_values))), 5))

    q_opt = optimistic_initialization(k, initial_value=5.0)
    print("Optimistic Q init:", q_opt[:3], "...")

    counts = np.ones(k)
    action = ucb_action_select(q_opt, counts, timestep=1, c=2.0)
    print("UCB first action:", int(action))

    prefs = np.zeros(k)
    prefs = gradient_bandit_update(prefs, action=0, reward=1.0, average_reward=0.5, alpha=0.1)
    print("Gradient prefs sample:", np.round(prefs[:3], 4))

    settings = [
        {"method": "epsilon_greedy", "param": 0.1},
        {"method": "optimistic", "param": 5.0},
        {"method": "ucb", "param": 2.0},
        {"method": "gradient", "param": 0.1},
    ]
    study = bandit_parameter_study(n_runs=30, n_steps=200, seed=0, settings=settings)
    print("Parameter study results:", study)

    # --- Gridworld MDP: policy & value iteration ---
    mdp = build_gridworld_mdp()
    gamma, theta = 0.9, 1e-4

    pi_values, pi_policy = policy_iteration(mdp, gamma=gamma, theta=theta)
    print("Policy iteration V[0]:", round(float(np.asarray(pi_values).ravel()[0]), 4))
    print("Policy iteration policy (flat):", np.asarray(pi_policy).ravel()[:5], "...")

    vi_values, vi_policy = value_iteration(mdp, gamma=gamma, theta=theta)
    print("Value iteration V[0]:", round(float(np.asarray(vi_values).ravel()[0]), 4))

    # --- Gambler's problem ---
    goal, head_prob = 100, 0.4
    g_values = gambler_value_iteration(goal, head_prob, theta=1e-6, gamma=1.0)
    stakes = extract_optimal_stakes(g_values, goal, head_prob, gamma=1.0)
    capitals = [1, 25, 50, 75, 99]
    print("Gambler V at", capitals, ":", [round(float(g_values[c]), 4) for c in capitals])
    print("Optimal stakes at", capitals, ":", [int(stakes[c]) for c in capitals])


if __name__ == "__main__":
    main()
