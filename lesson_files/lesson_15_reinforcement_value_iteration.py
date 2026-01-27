# %% [markdown]
# # Lesson 15 — Reinforcement Learning: Value Iteration in Gridworld
#
# Reinforcement learning deals with sequential decision making in environments modeled
# as Markov decision processes (MDPs).  In this notebook we implement **value
# iteration**, a dynamic programming algorithm that computes the optimal policy by
# iteratively improving estimates of the state value function.  We demonstrate on a
# simple gridworld where the agent seeks to reach a goal while avoiding obstacles.
#
# ## Outline
#
# - **MDP definition**: states, actions, transition probabilities and rewards.
# - **Value iteration algorithm**: Bellman optimality update.
# - **Policy extraction**: derive optimal actions from the value function.
# - **Visualization**: display the optimal value function and policy arrows.
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & Gridworld Setup

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Define gridworld dimensions
grid_size = (5, 5)

# Define rewards: -1 per step, 0 at goal, -10 at a pit state
goal_state = (4, 4)
pit_state = (2, 2)

def reward(state):
    if state == goal_state:
        return 0
    if state == pit_state:
        return -10
    return -1

# Possible actions: up, down, left, right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Transition model: deterministic for simplicity
def next_state(state, action):
    r, c = state
    dr, dc = action
    r_new = min(max(r + dr, 0), grid_size[0] - 1)
    c_new = min(max(c + dc, 0), grid_size[1] - 1)
    return (r_new, c_new)


# %% [markdown]
# ### Value Iteration Algorithm

def value_iteration(grid_size, gamma=0.9, theta=1e-4):
    """Compute the optimal value function using value iteration."""
    V = np.zeros(grid_size)
    iteration = 0
    while True:
        delta = 0
        for r in range(grid_size[0]):
            for c in range(grid_size[1]):
                state = (r, c)
                if state == goal_state:
                    continue
                v_old = V[state]
                values = []
                for action in actions:
                    ns = next_state(state, action)
                    v = reward(ns) + gamma * V[ns]
                    values.append(v)
                V[state] = max(values)
                delta = max(delta, abs(v_old - V[state]))
        iteration += 1
        if delta < theta:
            break
    return V

def extract_policy(V, gamma=0.9):
    """Extract the optimal policy from the value function."""
    policy = np.zeros((*grid_size, len(actions)))
    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            state = (r, c)
            if state == goal_state:
                continue
            values = []
            for action in actions:
                ns = next_state(state, action)
                v = reward(ns) + gamma * V[ns]
                values.append(v)
            best_action_idx = np.argmax(values)
            policy[r, c, best_action_idx] = 1
    return policy


# %% [markdown]
# ### Running Value Iteration and Visualizing Results

gamma = 0.9
V_opt = value_iteration(grid_size, gamma=gamma)
policy_opt = extract_policy(V_opt, gamma=gamma)

print("Optimal value function:")
print(V_opt)

# Plot value function heatmap
plt.figure(figsize=(5, 4))
plt.imshow(V_opt, cmap='coolwarm')
plt.colorbar(label="Value")
plt.title("Optimal State Value Function")
plt.scatter([pit_state[1]], [pit_state[0]], marker='x', color='black', s=100, label="Pit")
plt.scatter([goal_state[1]], [goal_state[0]], marker='*', color='gold', s=150, label="Goal")
plt.legend()
plt.show()

# Plot policy arrows
plt.figure(figsize=(5, 4))
plt.imshow(V_opt, cmap='coolwarm')
for r in range(grid_size[0]):
    for c in range(grid_size[1]):
        if (r, c) == goal_state:
            continue
        a_idx = np.argmax(policy_opt[r, c])
        dr, dc = actions[a_idx]
        plt.arrow(c, r, 0.3 * dc, -0.3 * dr, head_width=0.2, head_length=0.2, fc='k', ec='k')
plt.scatter([pit_state[1]], [pit_state[0]], marker='x', color='black', s=100, label="Pit")
plt.scatter([goal_state[1]], [goal_state[0]], marker='*', color='gold', s=150, label="Goal")
plt.title("Optimal Policy Arrows")
plt.legend()
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Discount Factor**: Vary \(\gamma\) and observe its impact on the value function and policy.
#    A smaller \(\gamma\) makes the agent more myopic.
# 2. **Stochastic Transitions**: Modify `next_state` to include randomness (e.g. 80%
#    probability of intended move, 20% random move).  Update value iteration accordingly.
# 3. **Policy Iteration**: Implement the policy iteration algorithm and compare its
#    convergence to value iteration.
# 4. **Larger Gridworld**: Increase the grid size, add more obstacles and rewards,
#    and visualize the optimal policy.
#
# ### Interview‑Ready Summary
#
# - Value iteration uses dynamic programming to compute the optimal state value function
#   by repeatedly applying the Bellman optimality operator until convergence.
# - The update rule considers the expected return of each action and chooses the
#   maximum, yielding the value of a state under the optimal policy.
# - Once the value function is computed, the optimal policy is extracted by choosing
#   the action that maximizes the expected return from each state.
# - The discount factor \(\gamma\) balances immediate vs. future rewards.  Stochastic
#   transitions require summing over next states weighted by transition probabilities.
# - Policy iteration alternates between policy evaluation and policy improvement and
#   often converges faster than value iteration.
