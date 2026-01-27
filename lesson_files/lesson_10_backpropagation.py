# %% [markdown]
# # Lesson 10 — Backpropagation in Depth
#
# Backpropagation is the workhorse algorithm that efficiently computes gradients in
# neural networks.  In this notebook we derive and implement backpropagation for a
# simple network with two hidden layers on the XOR problem, a classic example of a
# nonlinearly separable dataset.  We illustrate how the chain rule propagates
# gradients from the output back to earlier layers.
#
# ## Outline
#
# - **XOR dataset**: generate a minimal binary classification problem.
# - **Network architecture**: two hidden layers and a sigmoid activation.
# - **Forward pass**: compute activations layer by layer.
# - **Backpropagation**: derive gradients for all parameters.
# - **Training loop**: stochastic gradient descent on the tiny dataset.
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & Data Setup

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

print("XOR dataset:")
print(np.hstack([X, y]))


# %% [markdown]
# ### Network Architecture & Initialization

input_dim = 2
hidden_dim1 = 4
hidden_dim2 = 4
output_dim = 1

# Initialize weights and biases
W1 = np.random.randn(input_dim, hidden_dim1) * 0.5
b1 = np.zeros((1, hidden_dim1))
W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.5
b2 = np.zeros((1, hidden_dim2))
W3 = np.random.randn(hidden_dim2, output_dim) * 0.5
b3 = np.zeros((1, output_dim))


# %% [markdown]
# ### Activation Functions

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    return a * (1 - a)


# %% [markdown]
# ### Forward Pass

def forward_pass(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray):
    """Compute activations for all layers."""
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    Z3 = A2 @ W3 + b3
    A3 = sigmoid(Z3)  # final output in [0,1]
    cache = (Z1, A1, Z2, A2, Z3, A3)
    return A3, cache


# %% [markdown]
# ### Backpropagation
#
# We derive gradients using the chain rule.  Let the loss be the binary cross‑entropy:
# \(L = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)})]\).
# The derivative of the loss w.r.t. the output activation is \(dA3 = -(y / A3 - (1 - y)/(1 - A3))\).  We
# propagate this back through the network.

def backward_pass(X: np.ndarray, y_true: np.ndarray, cache, W1, W2, W3):
    """Compute gradients for all parameters via backpropagation."""
    Z1, A1, Z2, A2, Z3, A3 = cache
    m = X.shape[0]
    # Derivative of loss w.r.t. A3
    dA3 = - (y_true / (A3 + 1e-15) - (1 - y_true) / (1 - A3 + 1e-15)) / m
    # Layer 3
    dZ3 = dA3 * sigmoid_derivative(A3)
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)
    dA2 = dZ3 @ W3.T
    # Layer 2
    dZ2 = dA2 * sigmoid_derivative(A2)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = dZ2 @ W2.T
    # Layer 1
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3


# %% [markdown]
# ### Training Loop

learning_rate = 0.5
epochs = 10000

loss_history = []

for epoch in range(epochs):
    # Forward
    A3, cache = forward_pass(X, W1, b1, W2, b2, W3, b3)
    # Loss (binary cross‑entropy)
    loss = -np.mean(y * np.log(A3 + 1e-15) + (1 - y) * np.log(1 - A3 + 1e-15))
    loss_history.append(loss)
    # Backward
    dW1, db1, dW2, db2, dW3, db3 = backward_pass(X, y, cache, W1, W2, W3)
    # Update
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    # Optionally print progress
    if (epoch + 1) % 2000 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Plot loss curve
plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Binary Cross‑Entropy Loss")
plt.title("Training Loss for XOR Network")
plt.show()

# Final predictions
preds = (A3 >= 0.5).astype(int)
accuracy = (preds == y).mean()
print(f"Final training accuracy on XOR: {accuracy * 100:.2f}%")


# %% [markdown]
# ### Exercises
#
# 1. **Activation Functions**: Replace the sigmoid activation with ReLU in the hidden layers
#    and observe the effect on convergence and required hidden units.
# 2. **Network Depth**: Experiment with a single hidden layer vs. two hidden layers.
#    When does a deeper network perform better, and why is depth necessary for XOR?
# 3. **Learning Rate Scheduling**: Implement a decaying learning rate schedule and
#    compare convergence rates.
# 4. **Vectorization**: Extend this implementation to handle mini‑batches and larger
#    datasets.  Derive the gradients accordingly.
#
# ### Interview‑Ready Summary
#
# - Backpropagation applies the chain rule to efficiently compute gradients of a
#   network's parameters.  Each layer's gradient depends on the derivative of its
#   activation and the gradients of subsequent layers.
# - Multi‑layer networks can represent functions that are not linearly separable (e.g., XOR).
#   Depth enables compositional representations that capture complex patterns.
# - Binary cross‑entropy loss and sigmoid activations are appropriate for binary
#   classification; softmax and cross‑entropy generalize to multiclass.
# - Proper weight initialization and learning rate selection are critical for
#   convergence; too large a learning rate can cause divergence, while too small slows
#   learning.
