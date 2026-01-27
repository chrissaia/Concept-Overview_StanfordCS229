# %% [markdown]
# # Lesson 9 — Neural Networks & Backpropagation
#
# Neural networks are flexible function approximators composed of layers of linear
# transformations followed by nonlinear activation functions.  In this notebook we
# build a simple feedforward neural network from scratch using **NumPy** to classify
# handwritten digits.  We implement forward propagation, backpropagation and gradient
# descent to train the network on the `digits` dataset from scikit‑learn.
#
# ## Outline
#
# - **Data loading & preprocessing**: load digit images, flatten and normalize.
# - **Network architecture**: one hidden layer with nonlinear activation.
# - **Forward pass**: compute activations for hidden and output layers.
# - **Backpropagation**: derive gradients of weights and biases.
# - **Training loop**: update parameters via gradient descent.
# - **Evaluation & visualization**: report accuracy and show some predictions.
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & Data Preparation

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

np.random.seed(0)

# Load the handwritten digits dataset (64 features, 10 classes)
digits = datasets.load_digits()
X_raw = digits.data  # shape (1797, 64)
y_raw = digits.target.reshape(-1, 1)  # shape (1797, 1)

# Normalize features to [0, 1]
X = X_raw / 16.0

# One‑hot encode labels
encoder = OneHotEncoder(sparse=False, categories='auto')
y_onehot = encoder.fit_transform(y_raw)  # shape (1797, 10)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

print(f"Digits dataset: {X_train.shape[0]} training examples, {X_test.shape[0]} test examples.")


# %% [markdown]
# ### Network Architecture & Initialization
#
# We define a neural network with one hidden layer.  Let the input dimension be
# \(n_0 = 64\), hidden layer size \(n_1\), and output dimension \(n_2 = 10\).  The parameters
# consist of weight matrices \(W^{(1)} \in \mathbb{R}^{n_0 \times n_1}\) and \(W^{(2)} \in \mathbb{R}^{n_1 \times n_2}\),
# and bias vectors \(b^{(1)} \in \mathbb{R}^{n_1}\) and \(b^{(2)} \in \mathbb{R}^{n_2}\).  We use the hyperbolic
# tangent activation for the hidden layer and the softmax function for the output.

input_dim = X_train.shape[1]
hidden_dim = 32
output_dim = y_train.shape[1]

# Initialize weights with small random values and biases with zeros
W1 = 0.01 * np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
W2 = 0.01 * np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))


# %% [markdown]
# ### Activation Functions and Forward Pass

def softmax(z: np.ndarray) -> np.ndarray:
    """Compute softmax row‑wise in a numerically stable way."""
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute activations for the hidden and output layers."""
    Z1 = X @ W1 + b1  # hidden pre‑activation
    A1 = np.tanh(Z1)  # hidden activation
    Z2 = A1 @ W2 + b2  # output pre‑activation
    A2 = softmax(Z2)   # output probabilities
    return Z1, A1, Z2, A2


# %% [markdown]
# ### Loss Function and Backpropagation
#
# We use the multiclass cross‑entropy loss:
#
# \[
# J = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^{n_2} y^{(i)}_k \log \hat{y}^{(i)}_k,
# \]
#
# where \(\hat{y}^{(i)}\) are the softmax outputs.  The gradients can be derived using
# the chain rule.

def compute_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    m = y_true.shape[0]
    # add small epsilon to avoid log(0)
    loss = -np.sum(y_true * np.log(y_pred + 1e-15)) / m
    return float(loss)

def backward(X: np.ndarray, y_true: np.ndarray, Z1: np.ndarray, A1: np.ndarray, A2: np.ndarray, W2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute gradients of weights and biases via backpropagation."""
    m = X.shape[0]
    # Gradient of loss w.r.t. output pre‑activation
    dZ2 = (A2 - y_true) / m  # shape (m, n2)
    dW2 = A1.T @ dZ2         # shape (n1, n2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    # Gradient w.r.t. hidden layer
    dA1 = dZ2 @ W2.T             # shape (m, n1)
    dZ1 = dA1 * (1 - np.tanh(Z1) ** 2)  # derivative of tanh
    dW1 = X.T @ dZ1              # shape (n0, n1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2


# %% [markdown]
# ### Training Loop

learning_rate = 0.1
epochs = 200
batch_size = 64

loss_history = []

for epoch in range(epochs):
    # Mini‑batch gradient descent
    perm = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        # Forward
        Z1, A1, Z2, A2 = forward(X_batch, W1, b1, W2, b2)
        # Backward
        dW1, db1, dW2, db2 = backward(X_batch, y_batch, Z1, A1, A2, W2)
        # Update
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
    # Compute loss on full training set for monitoring
    _, _, _, A2_full = forward(X_train, W1, b1, W2, b2)
    loss = compute_loss(A2_full, y_train)
    loss_history.append(loss)
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Training loss: {loss:.4f}")

# Plot training loss
plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Cross‑Entropy Loss")
plt.title("Neural Network Training Loss")
plt.show()


# %% [markdown]
# ### Evaluating Accuracy & Predictions

def predict_nn(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Predict class labels for a dataset."""
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1).reshape(-1, 1)

# Training accuracy
y_train_pred = predict_nn(X_train, W1, b1, W2, b2)
y_train_true = np.argmax(y_train, axis=1).reshape(-1, 1)
train_accuracy = (y_train_pred == y_train_true).mean()

# Test accuracy
y_test_pred = predict_nn(X_test, W1, b1, W2, b2)
y_test_true = np.argmax(y_test, axis=1).reshape(-1, 1)
test_accuracy = (y_test_pred == y_test_true).mean()

print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Show a few test predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.ravel()):
    idx = np.random.randint(0, X_test.shape[0])
    ax.imshow(digits.images[digits.target == y_test_true[idx]][0], cmap='gray')  # show any sample of that digit
    ax.set_title(f"Pred: {int(y_test_pred[idx])}")
    ax.axis('off')
plt.suptitle("Sample Digit Predictions (labels may not match samples)")
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Hidden Layer Size**: Experiment with different numbers of hidden units and observe
#    how training and test accuracy change.  Does a larger network always perform better?
# 2. **Activation Functions**: Replace `tanh` with ReLU (rectified linear unit) or sigmoid and
#    modify the backpropagation accordingly.  Compare convergence and performance.
# 3. **Momentum & Regularization**: Implement momentum or Adam optimizer to accelerate
#    training.  Add \(\ell_2\) regularization to the weight gradients and observe its effect.
# 4. **Deep Networks**: Add another hidden layer and derive the backpropagation equations.
#
# ### Interview‑Ready Summary
#
# - A feedforward neural network is composed of layers of linear transforms followed by
#   nonlinear activation functions.  The weights and biases are trained to minimize a
#   loss function via gradient descent.
# - Backpropagation efficiently computes gradients by applying the chain rule from the
#   output layer backwards to the input.  Each layer's gradient depends on the
#   derivative of its activation function and the gradients of subsequent layers.
# - The softmax function converts raw scores into probabilities that sum to one.  It is
#   paired with the cross‑entropy loss for multiclass classification.
# - Proper initialization and normalization (e.g., dividing pixel values) help
#   accelerate convergence.  Choice of activation and network size affects capacity and
#   risk of overfitting.
# - More sophisticated optimizers (e.g., Adam) and regularization techniques can
#   significantly improve training speed and generalization.
