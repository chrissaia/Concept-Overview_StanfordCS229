# %% [markdown]
# # Lesson 18 — Policy Gradient Methods: REINFORCE on a Multi‑Armed Bandit
#
# Policy gradient methods optimize a parametrized policy directly by ascending the
# gradient of expected returns.  In this lesson we implement the REINFORCE algorithm
# on a simple multi‑armed bandit problem.  Although bandits have no state dynamics,
# they provide a clear illustration of how to update policy parameters using sampled
# rewards.
#
# ## Outline
#
# - **Bandit environment**: define arms with different reward probabilities.
# - **Policy parameterization**: softmax over preferences for each arm.
# - **REINFORCE algorithm**: update preferences using sampled rewards.
# - **Learning curves**: track the probability of selecting the optimal arm.
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & Bandit Definition

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Define a 3‑armed bandit: each arm yields reward 1 with a different probability
arm_probs = np.array([0.2, 0.5, 0.8])  # arm 2 is optimal
n_arms = len(arm_probs)

def pull_arm(arm):
    return 1 if np.random.rand() < arm_probs[arm] else 0


# %% [markdown]
# ### Policy Parameterization and REINFORCE
#
# We maintain a vector of preferences \(\theta\) over arms.  The policy \(\pi(a; \theta)\)
# is given by the softmax of preferences.  The REINFORCE update for a bandit is
#
# \[
# \theta \leftarrow \theta + \alpha \bigl(r - b\bigr) \nabla_\theta \log \pi(a; \theta),
# \]
#
# where \(r\) is the obtained reward and \(b\) is a baseline to reduce variance.

def softmax(prefs: np.ndarray) -> np.ndarray:
    z = prefs - prefs.max()
    exp_p = np.exp(z)
    return exp_p / exp_p.sum()

def reinforce_bandit(arm_probs, episodes=2000, alpha=0.1, baseline=True):
    n_arms = len(arm_probs)
    prefs = np.zeros(n_arms)
    avg_reward = 0.0  # baseline
    optimal_arm = np.argmax(arm_probs)
    optimal_probs = []
    for ep in range(1, episodes + 1):
        policy = softmax(prefs)
        # Sample action
        arm = np.random.choice(n_arms, p=policy)
        reward = pull_arm(arm)
        # Update baseline (running average of rewards)
        avg_reward += (reward - avg_reward) / ep
        # Gradient of log pi wrt prefs is (1 - pi(a_i)) for chosen arm i, -pi(j) for others
        grad_log_pi = -policy
        grad_log_pi[arm] += 1
        # Use baseline to reduce variance if enabled
        baseline_val = avg_reward if baseline else 0.0
        prefs += alpha * (reward - baseline_val) * grad_log_pi
        # Track probability of selecting optimal arm
        optimal_probs.append(policy[optimal_arm])
    return prefs, optimal_probs


# %% [markdown]
# ### Running REINFORCE and Plotting Results

episodes = 1000
prefs, opt_prob_history = reinforce_bandit(arm_probs, episodes=episodes, alpha=0.1, baseline=True)

print("Learned preferences:", prefs)
print("Learned policy (softmax of prefs):", softmax(prefs))

# Plot probability of choosing optimal arm over time
plt.figure(figsize=(6, 4))
plt.plot(opt_prob_history)
plt.xlabel("Episode")
plt.ylabel("Probability of choosing optimal arm")
plt.title("REINFORCE on a Multi‑Armed Bandit")
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Baselines**: Experiment with different baseline strategies: constant zero,
#    running average (as above), or state‑dependent baselines.  Compare variance and
#    convergence speed.
# 2. **Non‑Stationary Bandits**: Change the reward probabilities over time and observe
#    how the policy adapts.  Implement a sliding‑window baseline to adapt.
# 3. **Actor–Critic**: Extend this bandit example to an actor–critic method by
#    learning a separate value function (critic) to provide a baseline.
# 4. **Full MDP**: Implement REINFORCE on a simple episodic MDP (e.g., gridworld)
#    where trajectories are longer and involve state transitions.
#
# ### Interview‑Ready Summary
#
# - Policy gradient methods parameterize the policy and directly adjust parameters to
#   maximize expected returns using gradient ascent.  They are particularly suited
#   for problems with continuous or large action spaces.
# - The REINFORCE algorithm uses sampled returns to estimate the gradient of the
#   expected reward with respect to policy parameters.  It updates preferences in
#   the direction that increases the probability of actions yielding higher rewards.
# - Variance reduction via baselines (e.g., subtracting an estimate of the average
#   reward) improves learning stability.  Actor–critic methods generalize this idea
#   by learning a value function alongside the policy.
# - In bandit problems, REINFORCE learns to favour arms with higher reward
#   probabilities.  In full MDPs, it can learn stochastic policies that balance
#   exploration and exploitation over trajectories.
