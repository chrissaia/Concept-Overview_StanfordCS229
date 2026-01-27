# %% [markdown]
# # Lesson 2 — Linear Regression from Scratch
#
# In this notebook we will build a linear regression model from the ground up using only
# **NumPy**.  We follow the structure of CS229 to introduce notation, define the objective,
# derive the gradient descent algorithm, and build intuition through visualizations.  At the
# end you'll find exercises and interview‑style questions to reinforce understanding.
#
# ## Outline
#
# - **Introduction & Notation**: formalize the hypothesis, parameters and cost function.
# - **Data loading & preprocessing**: obtain a regression dataset and standardize features.
# - **Gradient Descent Implementation**: derive and implement batch gradient descent.
# - **Stochastic Gradient Descent**: implement SGD and compare convergence.
# - **Visualization**: plot the learned regressors on individual features.
# - **Exercises & Interview Q**: practice problems and summarization prompts.


# %% [markdown]
# ### Introduction & Notation
#
# We consider a supervised learning problem with training examples \(\{(x^{(i)}, y^{(i)})\}\_{i=1}^m\).  Each
# feature vector \(x^{(i)} \in \mathbb{R}^n\) is augmented with an intercept term to form \(\tilde{x}^{(i)} = [1, x_1^{(i)}, \dots, x_n^{(i)}]^T\).
# Our hypothesis family is linear:
#
# \[
# h_\theta(x) = \theta^T \tilde{x},
# \]
#
# where \(\theta \in \mathbb{R}^{n+1}\) collects the bias and weights.  The mean squared error (MSE) cost function
# over the dataset is
#
# \[
# J(\theta) = \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2.
# \]
#
# Minimizing \(J(\theta)\) with respect to \(\theta\) yields the ordinary least squares solution.  We will derive
# the gradient and use gradient descent to find \(\theta\).


# %% [markdown]
# ### Imports & Random Seed
# We'll use NumPy for array operations and Matplotlib for plotting.  We also import a toy
# dataset from scikit‑learn just to obtain a realistic regression problem.  Using `sklearn`
# to fetch data is allowed here, but we will **not** use its modeling APIs.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Fix the random seed for reproducibility
np.random.seed(0)


# %% [markdown]
# ### Data Loading & Preprocessing
#
# For demonstration purposes we use the **diabetes** dataset from scikit‑learn, which consists of
# ten real‑valued features and a continuous target.  We standardize each feature to have
# zero mean and unit variance.  Then we add an intercept column of ones.

# Load the dataset (only the features and target)
X_raw, y_raw = datasets.load_diabetes(return_X_y=True)

# Standardize features
X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
y = y_raw.reshape(-1, 1)  # ensure column vector

m, n = X.shape  # number of examples and features

# Add intercept term
Xb = np.hstack([np.ones((m, 1)), X])  # shape (m, n+1)

print(f"Loaded dataset with {m} examples and {n} features.")


# %% [markdown]
# ### Gradient Descent Derivation
#
# The partial derivative of the MSE cost with respect to \(\theta\) is
#
# \[
# \nabla_\theta J(\theta) = \frac{2}{m} X_b^T \bigl( X_b \theta - y \bigr).
# \]
#
# We initialize \(\theta\) to zeros and update it via the batch gradient descent rule:
#
# \[
# \theta := \theta - \alpha \nabla_\theta J(\theta),
# \]
#
# where \(\alpha\) is the learning rate.  Below we implement the loss function, its gradient,
# and a training loop to optimize \(\theta\).

def mse_loss(Xb: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Compute the mean squared error loss."""
    m = Xb.shape[0]
    errors = Xb @ theta - y
    return float((errors.T @ errors) / m)

def mse_gradient(Xb: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute the gradient of the MSE loss w.r.t. theta."""
    m = Xb.shape[0]
    return (2.0 / m) * Xb.T @ (Xb @ theta - y)

def gradient_descent(Xb: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 2000) -> tuple[np.ndarray, list]:
    """Perform batch gradient descent to minimize the MSE loss.

    Returns:
        theta (np.ndarray): optimized parameters
        history (list): list of loss values at each epoch
    """
    theta = np.zeros((Xb.shape[1], 1))
    history = []
    for epoch in range(epochs):
        loss = mse_loss(Xb, y, theta)
        history.append(loss)
        grad = mse_gradient(Xb, y, theta)
        theta -= lr * grad
    return theta, history


# %% [markdown]
# ### Training the Model
#
# We train the model using a moderate learning rate.  The convergence can be observed by
# plotting the loss over epochs.  In practice one might use more sophisticated learning
# rate schedules or optimize the number of epochs based on validation error.

learning_rate = 0.05
epochs = 500

theta_gd, history_gd = gradient_descent(Xb, y, lr=learning_rate, epochs=epochs)

print("Optimized theta shape:", theta_gd.shape)

# Plot loss over epochs
plt.figure(figsize=(6, 4))
plt.plot(history_gd)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Batch Gradient Descent Convergence")
plt.show()


# %% [markdown]
# ### Stochastic Gradient Descent
#
# In Stochastic Gradient Descent (SGD), we update the parameters using one training example
# at a time.  This can converge faster and allows online training on streaming data.  The
# update rule for a single sample \(i\) is:
#
# \[
# \theta := \theta - 2 \alpha \bigl( h_\theta(x^{(i)}) - y^{(i)} \bigr) \tilde{x}^{(i)}.
# \]
#
# We shuffle the training data each epoch to ensure the gradient direction varies over time.

def sgd(Xb: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 50, seed: int = 0) -> tuple[np.ndarray, list]:
    """Perform stochastic gradient descent."""
    rng = np.random.default_rng(seed)
    m = Xb.shape[0]
    theta = np.zeros((Xb.shape[1], 1))
    loss_history = []
    for epoch in range(epochs):
        indices = rng.permutation(m)
        for i in indices:
            xi = Xb[i:i+1]  # shape (1, n+1)
            yi = y[i:i+1]   # shape (1, 1)
            error = xi @ theta - yi
            theta -= lr * 2 * error * xi.T
        loss_history.append(mse_loss(Xb, y, theta))
    return theta, loss_history

theta_sgd, history_sgd = sgd(Xb, y, lr=0.01, epochs=100)

plt.figure(figsize=(6, 4))
plt.plot(history_sgd)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Stochastic Gradient Descent Convergence")
plt.show()

print("SGD final loss:", history_sgd[-1])


# %% [markdown]
# ### Visualization of Individual Features
#
# For interpretability we plot the learned regression line against the true data for a few
# selected features.  To keep plots readable we display the first four features.

feature_names = [f"Feature {i}" for i in range(n)]

plt.figure(figsize=(12, 8))
for idx in range(4):
    Xi = X[:, idx].reshape(-1, 1)
    # Build line for plotting
    x_line = np.linspace(Xi.min(), Xi.max(), 200).reshape(-1, 1)
    Xb_line = np.zeros((len(x_line), Xb.shape[1]))
    Xb_line[:, 0] = 1
    Xb_line[:, idx + 1] = x_line[:, 0]
    y_line = Xb_line @ theta_gd

    plt.subplot(2, 2, idx + 1)
    plt.scatter(Xi, y, alpha=0.2)
    plt.plot(x_line, y_line, color="red")
    plt.xlabel(feature_names[idx])
    plt.ylabel("Target")
    plt.title(f"Linear Fit for {feature_names[idx]}")
plt.suptitle("Model Fits for Selected Features")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Normal equation derivation**: Show that the optimal \(\theta\) minimizing the MSE has a closed form
#    \(\theta = (X_b^T X_b)^{-1} X_b^T y\).  Implement this formula and compare it to the gradient descent
#    result.
# 2. **Learning rate tuning**: Experiment with different learning rates and epochs for batch gradient descent.
#    Plot the learning curves to see how they influence convergence.
# 3. **Feature selection**: Use only a subset of features (e.g. top 2 principal components) and retrain the model.
#    How does feature dimensionality affect performance?
# 4. **Regularization**: Add an \(\ell_2\) regularization term \(\lambda \lVert \theta \rVert^2\) to the cost function
#    and derive the new gradient update.  Implement gradient descent with regularization.
#
# ### Interview‑Ready Summary
#
# - The linear regression model assumes a linear relationship between features and the response:
#   \(h_\theta(x) = \theta^T \tilde{x}\).
# - The mean squared error cost is convex; its gradient is \(\nabla J(\theta) = \frac{2}{m} X_b^T (X_b \theta - y)\).
# - **Gradient descent** updates parameters by taking steps opposite the gradient: \(\theta := \theta - \alpha \nabla J(\theta)\).
# - **Batch vs. Stochastic**: batch uses all data per update; stochastic uses one sample.  SGD often converges
#   faster but has higher variance in updates.
# - Adding an intercept term allows the model to fit a bias.  Standardizing features improves conditioning of the
#   problem.
# - Closed‑form solution exists via the normal equation, but iterative methods scale better to large datasets.
