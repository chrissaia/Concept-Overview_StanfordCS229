# %% [markdown]
# # Lesson 4 — The Perceptron Algorithm
#
# In this notebook we implement the classic perceptron algorithm for binary
# classification.  We explore how the perceptron finds a separating hyperplane
# for linearly separable data and compare it conceptually to logistic regression.
#
# ## Outline
#
# - **Problem setup & data**: choose a linearly separable subset of a dataset.
# - **Algorithm derivation**: perceptron update rule.
# - **Implementation**: train the perceptron and track mistakes.
# - **Visualization**: plot the decision boundary in two dimensions.
# - **Exercises & interview summary**: deeper questions and key points.


# %% [markdown]
# ### Imports & Data Preparation
#
# We'll use the Iris dataset but restrict to the first two classes (setosa and versicolor)
# to make the problem binary and linearly separable.  We select the first two features
# for visualization.  Labels are mapped to \(\{-1, +1\}\) to simplify the update rule.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(0)

# Load Iris dataset
iris = datasets.load_iris()
X_raw = iris.data[:, :2]  # take first two features for plotting
y_raw = iris.target

# Keep only classes 0 and 1 (setosa vs versicolor)
mask = y_raw < 2
X = X_raw[mask]
y = y_raw[mask]

# Map labels {0,1} to {-1,1}
y = np.where(y == 0, -1, 1).reshape(-1, 1)

# Add intercept term
m, n = X.shape
Xb = np.hstack([np.ones((m, 1)), X])

print(f"Perceptron dataset: {m} examples, {n} features (plus bias).")


# %% [markdown]
# ### Perceptron Update Rule
#
# The perceptron seeks a weight vector \(\theta\) such that \(\text{sign}(\theta^T \tilde{x}) = y\).  The
# algorithm initializes \(\theta\) and iteratively scans through the dataset, updating \(\theta\) whenever
# a misclassification occurs.  For an example \((\tilde{x}^{(i)}, y^{(i)})\), the update is:
#
# \[
# \text{if } y^{(i)} (\theta^T \tilde{x}^{(i)}) \le 0:\quad \theta := \theta + y^{(i)} \tilde{x}^{(i)}.
# \]
#
# For linearly separable data, the perceptron converges in a finite number of mistakes.

def perceptron_train(Xb: np.ndarray, y: np.ndarray, epochs: int = 50) -> tuple[np.ndarray, list]:
    """Train perceptron weights via the perceptron algorithm.

    Returns:
        theta: learned weight vector
        mistakes_history: list recording cumulative mistakes per epoch
    """
    theta = np.zeros((Xb.shape[1], 1))
    mistakes_history = []
    total_mistakes = 0
    for epoch in range(epochs):
        mistakes = 0
        for i in range(Xb.shape[0]):
            xi = Xb[i:i+1]  # shape (1, n+1)
            yi = y[i, 0]
            if yi * float(xi @ theta) <= 0:
                theta += (yi * xi).T
                mistakes += 1
        total_mistakes += mistakes
        mistakes_history.append(total_mistakes)
    return theta, mistakes_history


# %% [markdown]
# ### Training the Perceptron

epochs = 20
theta_perc, mistakes_history = perceptron_train(Xb, y, epochs=epochs)

print(f"Total mistakes made: {mistakes_history[-1]}")

# Plot cumulative mistakes over epochs
plt.figure(figsize=(6, 4))
plt.plot(mistakes_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Cumulative Mistakes")
plt.title("Perceptron Training Progress")
plt.show()


# %% [markdown]
# ### Decision Boundary Visualization
#
# Because we use two features, we can visualize the separating hyperplane in the feature
# space.  The learned decision boundary satisfies \(\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0\).

# Compute decision boundary line
theta_0, theta_1, theta_2 = theta_perc.flatten()
x_vals = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
if abs(theta_2) > 1e-6:
    y_vals = - (theta_0 + theta_1 * x_vals) / theta_2
else:
    y_vals = np.zeros_like(x_vals)

# Plot data points and decision boundary
plt.figure(figsize=(6, 5))
plt.scatter(X[y[:, 0] == -1][:, 0], X[y[:, 0] == -1][:, 1], label="Class -1", alpha=0.6)
plt.scatter(X[y[:, 0] == 1][:, 0], X[y[:, 0] == 1][:, 1], label="Class +1", alpha=0.6)
plt.plot(x_vals, y_vals, 'r-', label="Perceptron boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Decision Boundary")
plt.legend()
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Kernel Perceptron**: Extend the perceptron to non‑linearly separable data using the kernel
#    trick.  Implement a kernel perceptron with a polynomial or RBF kernel.
# 2. **Learning Rate**: Introduce a learning rate \(\alpha\) in the update rule: \(\theta := \theta + \alpha y^{(i)} \tilde{x}^{(i)}\).
#    Experiment with different \(\alpha\) values and observe the convergence behaviour.
# 3. **Feature Scaling**: Try normalizing the features before training.  Does it change the
#    number of mistakes or convergence speed?
# 4. **Comparison with Logistic Regression**: Train logistic regression on the same dataset.
#    Compare decision boundaries and classification accuracy.
#
# ### Interview‑Ready Summary
#
# - The perceptron algorithm is an online learning method that finds a linear separator
#   for linearly separable data by updating weights whenever a misclassification occurs.
# - It converges in a finite number of updates if a separating hyperplane exists.  The
#   convergence proof shows the number of mistakes is bounded by the margin and the
#   radius of the data.
# - Unlike logistic regression, the perceptron uses a discrete step function and does
#   not output probabilities; it is sensitive to feature scaling but easy to implement.
# - Introducing kernels allows the perceptron to classify non‑linearly separable data
#   by implicitly mapping inputs into higher‑dimensional feature spaces.
