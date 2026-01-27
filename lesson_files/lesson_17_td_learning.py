# %% [markdown]
# # Lesson 17 — Temporal Difference Learning with Function Approximation
#
# Temporal Difference (TD) learning combines ideas from Monte Carlo and dynamic
# programming methods to learn value functions from experience.  In this lesson we
# implement TD(0) with linear function approximation on a simple random walk
# environment.  We illustrate how the value estimates converge to the true values
# over episodes.
#
# ## Outline
#
# - **Environment**: random walk with terminal states.
# - **True value function**: analytical solution for comparison.
# - **Feature representation**: linear basis functions for states.
# - **TD(0) algorithm**: update rule for weight vector.
# - **Learning curve**: track root mean squared value error (RMSVE).
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & Environment Setup

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Random walk environment: states 0 (terminal) through 5 (terminal), states 1-4 are nonterminal
nonterminal_states = [1, 2, 3, 4]
terminal_states = [0, 5]

# True state values for nonterminal states (expected return starting from state i)
true_values = {1: 1/6, 2: 2/6, 3: 3/6, 4: 4/6}

# Transition probabilities: move left or right with equal probability
def step(state):
    if state in terminal_states:
        return state, 0
    move = np.random.choice([-1, 1])
    next_state = state + move
    # Reward is 1 only if we transition into state 5, else 0
    reward = 1 if next_state == 5 else 0
    return next_state, reward


# %% [markdown]
# ### Feature Representation & TD(0)
#
# We use a simple linear feature representation for each state: a binary feature for
# each nonterminal state.  The value estimate is \(\hat{v}(s; w) = w^T x(s)\).  TD(0)
# update for weight vector \(w\) when transitioning from state \(s\) to \(s'\) with reward
# \(r\) is
#
# \[
# w := w + \alpha \bigl[r + \gamma \hat{v}(s'; w) - \hat{v}(s; w)\bigr] x(s),
# \]
#
# where \(\gamma\) is the discount factor (1 for the random walk).

num_states = 5  # nonterminal states 1..4 -> we use index 1..4 inclusive
feature_dim = 5

# One‑hot feature representation: x(s) has 1 at index s and 0 elsewhere
def features(state):
    x = np.zeros(feature_dim)
    if state in nonterminal_states:
        x[state] = 1
    return x

def td_learning(alpha=0.1, episodes=100):
    w = np.zeros(feature_dim)
    rmsve_history = []
    gamma = 1.0
    for ep in range(episodes):
        state = np.random.choice(nonterminal_states)
        while state not in terminal_states:
            x_s = features(state)
            next_state, reward = step(state)
            x_s_next = features(next_state)
            # TD target
            target = reward + gamma * np.dot(w, x_s_next)
            prediction = np.dot(w, x_s)
            # Update weights
            w += alpha * (target - prediction) * x_s
            state = next_state
        # Compute RMSVE after each episode
        squared_error = 0.0
        for s in nonterminal_states:
            v_hat = np.dot(w, features(s))
            squared_error += (true_values[s] - v_hat) ** 2
        rmsve = np.sqrt(squared_error / len(nonterminal_states))
        rmsve_history.append(rmsve)
    return w, rmsve_history


# %% [markdown]
# ### Running TD(0) and Plotting Convergence

alpha = 0.05
episodes = 100
weights, rmsve_hist = td_learning(alpha=alpha, episodes=episodes)

print("Learned weights:", weights)

# Plot RMS value error over episodes
plt.figure(figsize=(6, 4))
plt.plot(rmsve_hist)
plt.xlabel("Episode")
plt.ylabel("RMS Value Error")
plt.title(f"TD(0) Convergence (alpha={alpha})")
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Step Size Sensitivity**: Run TD(0) with different learning rates \(\alpha\) and compare
#    convergence speeds and stability.  Is there an optimal \(\alpha\)?
# 2. **Different Feature Representations**: Try using polynomial features (e.g., state index
#    normalized between 0 and 1) or coarse coding.  Compare the approximation
#    accuracy.
# 3. **On‑Policy vs. Off‑Policy**: Implement TD learning for the random walk under
#    different behaviour policies and targets (e.g., importance sampling).
# 4. **TD(λ)**: Extend the algorithm to TD(\(\lambda\)) with eligibility traces and study
#    its effect on bias and variance.
#
# ### Interview‑Ready Summary
#
# - TD learning updates value estimates based on bootstrapped targets, combining
#   Monte Carlo sampling with dynamic programming.  It can learn online from
#   experience without waiting for an episode to terminate.
# - With linear function approximation, the TD(0) update adjusts the weight vector
#   in the direction of the temporal difference error multiplied by the feature
#   vector: \(w \leftarrow w + \alpha (r + \gamma v(s') - v(s)) x(s)\).
# - Choosing an appropriate step size \(\alpha\) is crucial: too large leads to
#   divergence, while too small slows learning.
# - Feature representation determines the class of value functions that can be
#   approximated; richer features can reduce approximation error but may require
#   more data to train.
# - TD methods extend naturally to control via actor–critic and Q‑learning algorithms
#   by learning action values and policies simultaneously.
