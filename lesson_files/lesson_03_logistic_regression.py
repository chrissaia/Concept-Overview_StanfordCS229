# %% [markdown]
# # Lesson 3 — Logistic Regression from Scratch
#
# In this notebook we derive and implement logistic regression using only **NumPy**.  We
# discuss the underlying probabilistic interpretation, derive the gradient of the
# cross‑entropy loss, and implement gradient descent to fit the model.  We'll work on
# a binary classification dataset to illustrate the decision boundary and measure
# performance.
#
# ## Outline
#
# - **Problem setup & notation**: binary classification, logistic hypothesis and cost.
# - **Data loading & preprocessing**: load a real dataset and standardize features.
# - **Gradient descent implementation**: derive and code the update rule.
# - **Model evaluation**: compute accuracy and visualize the decision boundary.
# - **Exercises & interview summary**: practice questions and key takeaways.


# %% [markdown]
# ### Problem Setup & Notation
#
# Consider a dataset \(\{(x^{(i)}, y^{(i)})\}\_{i=1}^m\) where \(y^{(i)} \in \{0,1\}\) indicates class
# membership.  We augment each feature vector with a 1 to obtain \(\tilde{x}^{(i)} \in \mathbb{R}^{n+1}\).
# The logistic regression hypothesis uses the sigmoid function:
#
# \[
# h_\theta(x) = \sigma(\theta^T \tilde{x}) = \frac{1}{1 + e^{-\theta^T \tilde{x}}}.
# \]
#
# The cost function is the negative log‑likelihood (cross‑entropy):
#
# \[
# J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log (1 - h_\theta(x^{(i)}))\right].
# \]
#
# Differentiating \(J\) with respect to \(\theta\) yields the gradient
#
# \[
# \nabla_\theta J(\theta) = \frac{1}{m} X_b^T \bigl(h_\theta(X) - y\bigr),
# \]
# where \(X_b\) is the design matrix with an intercept column.  We will implement gradient
# descent to minimize this cost.


# %% [markdown]
# ### Imports & Random Seed

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(0)


# %% [markdown]
# ### Data Loading & Preprocessing
#
# We use the **breast cancer** dataset from `sklearn.datasets`, a binary classification problem with
# 30 features.  We standardize the features to have zero mean and unit variance and map
# the labels to 0 and 1.  Then we add an intercept term.

X_raw, y_raw = datasets.load_breast_cancer(return_X_y=True)

# Standardize features
X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
y = y_raw.reshape(-1, 1)

# Number of examples and features
m, n = X.shape

# Add intercept column
Xb = np.hstack([np.ones((m, 1)), X])  # shape (m, n+1)

print(f"Dataset: {m} examples, {n} features (after standardization).")


# %% [markdown]
# ### Logistic Hypothesis, Loss and Gradient Functions

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function elementwise."""
    return 1 / (1 + np.exp(-z))

def log_loss(Xb: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Compute the cross‑entropy loss for logistic regression."""
    m = Xb.shape[0]
    h = sigmoid(Xb @ theta)
    # Clip h to avoid log(0)
    h = np.clip(h, 1e-15, 1 - 1e-15)
    loss = - (y * np.log(h) + (1 - y) * np.log(1 - h)).mean()
    return float(loss)

def log_gradient(Xb: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute the gradient of the logistic loss."""
    m = Xb.shape[0]
    h = sigmoid(Xb @ theta)
    return (1.0 / m) * Xb.T @ (h - y)

def gradient_descent_logistic(Xb: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 2000) -> tuple[np.ndarray, list]:
    """Perform gradient descent for logistic regression."""
    theta = np.zeros((Xb.shape[1], 1))
    history = []
    for epoch in range(epochs):
        loss = log_loss(Xb, y, theta)
        history.append(loss)
        grad = log_gradient(Xb, y, theta)
        theta -= lr * grad
    return theta, history


# %% [markdown]
# ### Training the Logistic Regression Model

learning_rate = 0.01
epochs = 500

theta_lr, history_lr = gradient_descent_logistic(Xb, y, lr=learning_rate, epochs=epochs)

print("Final loss:", history_lr[-1])

# Plot loss over epochs
plt.figure(figsize=(6, 4))
plt.plot(history_lr)
plt.xlabel("Epoch")
plt.ylabel("Cross‑Entropy Loss")
plt.title("Logistic Regression Convergence")
plt.show()


# %% [markdown]
# ### Model Evaluation
#
# To evaluate the classifier we predict labels using a 0.5 threshold on \(h_\theta(x)\).
# We compute the accuracy and visualize the decision boundary using the first two features for
# illustration.  Note that the true classification uses all features, but we can project
# onto two dimensions for plotting.

def predict(Xb: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Predict binary labels using the learned parameters."""
    probs = sigmoid(Xb @ theta)
    return (probs >= 0.5).astype(int)

preds = predict(Xb, theta_lr)
accuracy = (preds == y).mean()
print(f"Training accuracy: {accuracy * 100:.2f}%")

# Visualization using two features (indices 0 and 1)
feat1, feat2 = 0, 1
X2 = X[:, [feat1, feat2]]
X2b = np.hstack([np.ones((m, 1)), X2])

# Compute line parameters for decision boundary using only two features
theta_2d = theta_lr[[0, feat1 + 1, feat2 + 1]]  # intercept + selected weights
# Decision boundary: theta_0 + theta_1 x1 + theta_2 x2 = 0 => x2 = -(theta_0 + theta_1 x1)/theta_2
x_vals = np.linspace(X2[:, 0].min(), X2[:, 0].max(), 200)
if abs(theta_2d[2, 0]) > 1e-6:
    y_vals = - (theta_2d[0, 0] + theta_2d[1, 0] * x_vals) / theta_2d[2, 0]
else:
    y_vals = np.zeros_like(x_vals)

# Plot
plt.figure(figsize=(6, 5))
plt.scatter(X2[y[:, 0] == 0][:, 0], X2[y[:, 0] == 0][:, 1], alpha=0.5, label="Class 0")
plt.scatter(X2[y[:, 0] == 1][:, 0], X2[y[:, 0] == 1][:, 1], alpha=0.5, label="Class 1")
plt.plot(x_vals, y_vals, 'r-', label="Decision boundary (2D projection)")
plt.xlabel(f"Feature {feat1}")
plt.ylabel(f"Feature {feat2}")
plt.title("Logistic Regression Decision Boundary (First Two Features)")
plt.legend()
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Newton's Method**: Derive the Hessian for logistic regression and implement Newton's
#    method for optimization.  Compare its convergence to gradient descent.
# 2. **Multiclass Logistic Regression**: Extend the model to handle multiple classes via
#    softmax and cross‑entropy loss.  Implement gradient descent for the multiclass case.
# 3. **Regularization**: Add \(\ell_2\) or \(\ell_1\) regularization to the cost function and study its
#    effect on the weight vector and decision boundary.
# 4. **Feature Engineering**: Explore adding polynomial terms or interactions between
#    features.  How does this change the classification accuracy?
#
# ### Interview‑Ready Summary
#
# - Logistic regression models the conditional probability \(P(y=1 \mid x)\) using the
#   sigmoid of a linear combination of features.
# - The cross‑entropy loss is convex; its gradient is the difference between predicted
#   probabilities and labels scaled by the features: \(\nabla J = \frac{1}{m} X_b^T(h - y)\).
# - Gradient descent iteratively updates parameters by subtracting the gradient scaled by
#   the learning rate.  A good choice of learning rate and number of epochs is critical
#   for convergence.
# - Thresholding the output at 0.5 yields binary predictions; accuracy is a common metric
#   but can be complemented by precision, recall and F1 score for imbalanced data.
# - Logistic regression can be regularized to prevent overfitting and extended to
#   multiclass problems via the softmax function.
