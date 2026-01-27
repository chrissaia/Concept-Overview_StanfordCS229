# %% [markdown]
# # Lesson 16 — Reinforcement Learning: Q‑Learning
#
# Q‑learning is a model‑free reinforcement learning algorithm that learns the value
# of taking a particular action in a given state, \(Q(s,a)\), without requiring
# knowledge of the environment's transition dynamics.  In this notebook we implement
# Q‑learning on a simple gridworld.  The agent learns optimal behaviour through
# exploration and exploitation.
#
# ## Outline
#
# - **Environment setup**: gridworld with rewards and terminal states.
# - **Q‑learning algorithm**: update rule and epsilon‑greedy exploration.
# - **Training loop**: run episodes and update the Q‑table.
# - **Visualization**: plot episode rewards and derived policy.
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & Environment Setup

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Gridworld parameters (reuse from Lesson 15)
grid_size = (5, 5)
goal_state = (4, 4)
pit_state = (2, 2)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

def step(state, action):
    """Take an action and return next state and reward."""
    r, c = state
    dr, dc = action
    r_new = min(max(r + dr, 0), grid_size[0] - 1)
    c_new = min(max(c + dc, 0), grid_size[1] - 1)
    next_state = (r_new, c_new)
    if next_state == pit_state:
        return next_state, -10
    elif next_state == goal_state:
        return next_state, 0
    else:
        return next_state, -1


# %% [markdown]
# ### Q‑Learning Algorithm

def q_learning(grid_size, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2):
    """Run Q‑learning on the gridworld and return the learned Q‑values and reward history."""
    Q = np.zeros((*grid_size, len(actions)))
    reward_history = []
    for ep in range(episodes):
        state = (0, 0)
        total_reward = 0
        while state != goal_state and state != pit_state:
            # Epsilon‑greedy action selection
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(len(actions))
            else:
                action_idx = np.argmax(Q[state])
            action = actions[action_idx]
            next_state, reward = step(state, action)
            total_reward += reward
            # Q‑update
            best_next = np.max(Q[next_state])
            Q[state + (action_idx,)] += alpha * (reward + gamma * best_next - Q[state + (action_idx,)])
            state = next_state
        reward_history.append(total_reward)
    return Q, reward_history


# %% [markdown]
# ### Training the Agent and Visualizing

episodes = 200
Q, rewards = q_learning(grid_size, episodes=episodes, alpha=0.1, gamma=0.9, epsilon=0.2)

print(f"Average reward over episodes: {np.mean(rewards):.2f}")

# Plot episode rewards
plt.figure(figsize=(6, 4))
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q‑Learning Training Rewards")
plt.show()

# Derive policy from Q values
policy = np.zeros((*grid_size, len(actions)))
for r in range(grid_size[0]):
    for c in range(grid_size[1]):
        if (r, c) == goal_state:
            continue
        best_action = np.argmax(Q[r, c])
        policy[r, c, best_action] = 1

# Visualize policy arrows
plt.figure(figsize=(5, 4))
value_function = np.max(Q, axis=2)
plt.imshow(value_function, cmap='coolwarm')
for r in range(grid_size[0]):
    for c in range(grid_size[1]):
        if (r, c) == goal_state:
            continue
        a_idx = np.argmax(policy[r, c])
        dr, dc = actions[a_idx]
        plt.arrow(c, r, 0.3 * dc, -0.3 * dr, head_width=0.2, head_length=0.2, fc='k', ec='k')
plt.scatter([pit_state[1]], [pit_state[0]], marker='x', color='black', s=100, label="Pit")
plt.scatter([goal_state[1]], [goal_state[0]], marker='*', color='gold', s=150, label="Goal")
plt.title("Derived Policy from Q‑Values")
plt.legend()
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Exploration Rate**: Experiment with different epsilon schedules (e.g., decay over time).
#    How does exploration affect learning speed and quality of the learned policy?
# 2. **Learning Rate**: Adjust \(\alpha\) and observe its effect on convergence and stability.
# 3. **Larger Environment**: Scale up the gridworld, add more obstacles and varied rewards.
#    Does Q‑learning still converge within a reasonable number of episodes?
# 4. **Function Approximation**: Replace the Q‑table with a neural network approximator and
#    implement Deep Q‑Learning (DQN) on a small environment.
#
# ### Interview‑Ready Summary
#
# - Q‑learning learns the action‑value function \(Q(s,a)\) by iteratively updating estimates
#   based on observed rewards and estimated value of the next state.  It does not
#   require a model of the environment's transitions.
# - The update rule is: \(Q(s,a) \leftarrow Q(s,a) + \alpha \bigl[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\bigr]\).
# - Exploration–exploitation trade‑off is typically handled via epsilon‑greedy
#   policies that choose a random action with probability \(\epsilon\) and the best
#   known action otherwise.
# - The learned policy is derived by selecting the action with the maximum Q‑value in
#   each state.  Visualization of Q‑values as a heatmap helps interpret the agent's
#   expectations.
# - Q‑learning can be extended to continuous state spaces and high‑dimensional
#   problems through function approximation (e.g., neural networks in Deep Q‑Learning).
