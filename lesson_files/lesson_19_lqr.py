# %% [markdown]
# # Lesson 19 — Linear Quadratic Regulator (LQR)
#
# The Linear Quadratic Regulator (LQR) solves continuous state‐space control problems
# with linear dynamics and quadratic cost.  It computes an optimal feedback gain
# matrix that minimizes the expected infinite‑horizon cost.  In this notebook we
# derive and implement the discrete‑time LQR solution, simulate the controlled
# system and analyze the resulting trajectories.
#
# ## Outline
#
# - **Problem formulation**: linear dynamics, quadratic cost.
# - **Riccati equation**: derive the discrete Riccati difference equation.
# - **Optimal feedback gain**: compute steady‑state solution.
# - **Simulation**: apply control law to the system and plot trajectories.
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & System Definition

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Define a simple 1D linear system: x_{t+1} = a x_t + b u_t
a = 1.1
b = 0.5
A = np.array([[a]])
B = np.array([[b]])

# Cost function: x^T Q x + u^T R u
Q = np.array([[1.0]])
R = np.array([[0.1]])

# Discount factor (for infinite horizon) – set to 1 for undiscounted LQR
gamma = 1.0


# %% [markdown]
# ### Riccati Equation & Optimal Gain
#
# The optimal feedback control law for the discrete‑time LQR is \(u = -K x\), where
# \(K\) is computed from the solution \(P\) of the algebraic Riccati equation:
#
# \[
# P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A.
# \]
#
# We iterate the Riccati difference equation until convergence to obtain \(P\).

def solve_discrete_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, gamma: float = 1.0, tol: float = 1e-8, max_iters: int = 1000):
    """Solve the discrete Riccati equation for LQR."""
    P = Q.copy()
    for _ in range(max_iters):
        K = np.linalg.inv(R + gamma * B.T @ P @ B) @ (gamma * B.T @ P @ A)
        P_next = Q + gamma * A.T @ P @ A - gamma * A.T @ P @ B @ K
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next
    K_opt = np.linalg.inv(R + gamma * B.T @ P @ B) @ (gamma * B.T @ P @ A)
    return P, K_opt


# Solve for P and K
P, K = solve_discrete_lqr(A, B, Q, R, gamma)
print("Optimal gain K:", K.flatten())


# %% [markdown]
# ### Simulating the Closed‑Loop System

def simulate_lqr(A, B, K, x0, T=30):
    """Simulate the closed‑loop linear system with feedback u = -Kx."""
    x = np.zeros((T + 1, A.shape[0]))
    u = np.zeros((T, B.shape[1]))
    x[0] = x0
    for t in range(T):
        u[t] = -K @ x[t]
        x[t + 1] = A @ x[t] + B @ u[t]
    return x, u

# Initial state
x0 = np.array([5.0])

# Simulate
T = 20
x_traj, u_traj = simulate_lqr(A, B, K, x0, T)

# Plot state and control trajectories
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x_traj)
plt.xlabel("Time step")
plt.ylabel("State x")
plt.title("State Trajectory under LQR Control")

plt.subplot(1, 2, 2)
plt.plot(u_traj)
plt.xlabel("Time step")
plt.ylabel("Control u")
plt.title("Control Trajectory")

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Multi‑Dimensional Systems**: Extend the example to higher‑dimensional systems (e.g.,
#    cart–pole linearization).  Derive the corresponding A and B matrices and solve
#    for the optimal feedback gain.
# 2. **Discounting**: Introduce \(\gamma < 1\) into the Riccati iteration and observe how the
#    gain and state trajectories change.
# 3. **Finite Horizon**: For a finite horizon T, implement the backward dynamic
#    programming recursion to compute time‑varying gains \(K_t\).
# 4. **Continuous‑Time LQR**: Derive and implement the continuous‑time LQR solution
#    using the continuous Riccati differential equation.
#
# ### Interview‑Ready Summary
#
# - LQR addresses control of linear dynamical systems with quadratic costs, yielding
#   closed‑form optimal feedback gains computed via the solution of a Riccati equation.
# - The discrete algebraic Riccati equation can be solved iteratively; the resulting
#   gain matrix produces a stabilizing controller that minimizes the infinite‑horizon cost.
# - The cost matrices Q and R weight state deviations and control effort, respectively.
#   Tuning Q and R trades off between accuracy (tracking) and energy usage.
# - LQR extends naturally to multi‑dimensional systems, continuous time and finite
#   horizons, and forms the basis of more advanced control methods such as LQG
#   (Linear Quadratic Gaussian) and MPC (Model Predictive Control).
