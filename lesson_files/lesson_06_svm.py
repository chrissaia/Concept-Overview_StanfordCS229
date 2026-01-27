# %% [markdown]
# # Lesson 6 — Support Vector Machines (SVM)
#
# Support Vector Machines are powerful margin‑based classifiers.  In their linear form
# they seek a hyperplane that maximizes the margin between classes while allowing for
# some misclassifications controlled by a regularization term.  In this notebook we
# implement a **linear soft‑margin SVM** using a subgradient descent approach on the
# hinge loss.  We use a binary dataset and compare the resulting classifier to
# logistic regression.
#
# ## Outline
#
# - **Data preparation**: load and preprocess a binary classification dataset.
# - **Hinge loss & gradient**: define the SVM objective and compute subgradients.
# - **Training via subgradient descent**: update rule with regularization.
# - **Model evaluation**: calculate accuracy and visualize projections.
# - **Exercises & interview summary**: deeper exploration and key takeaways.


# %% [markdown]
# ### Imports & Data Loading

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(0)

# Load the breast cancer dataset again (two classes)
X_raw, y_raw = datasets.load_breast_cancer(return_X_y=True)

# Standardize features
X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)

# Map labels to {-1, +1}
y = np.where(y_raw == 0, -1, 1).reshape(-1, 1)

# Add intercept term
m, n = X.shape
Xb = np.hstack([np.ones((m, 1)), X])

print(f"SVM dataset: {m} examples, {n} features (+ bias)")


# %% [markdown]
# ### Hinge Loss and Subgradient
#
# For a linear SVM with parameters \(\theta\) (including bias), the objective we minimize is
#
# \[
# L(\theta) = \frac{1}{m} \sum_{i=1}^m \max\bigl(0, 1 - y^{(i)} (\theta^T \tilde{x}^{(i)})\bigr)
# + \frac{\lambda}{2} \lVert \theta \rVert^2,
# \]
#
# where \(\lambda\) controls the trade‑off between margin maximization and slack penalties.  The
# subgradient of the hinge loss is
#
# \[
# \partial L = - \frac{1}{m} \sum_{i: y^{(i)} \theta^T \tilde{x}^{(i)} < 1} y^{(i)} \tilde{x}^{(i)} + \lambda \theta.
# \]
#
# We will implement a basic subgradient descent loop to minimize this objective.

def svm_subgradient(Xb: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_: float) -> np.ndarray:
    """Compute the subgradient of the SVM objective."""
    m = Xb.shape[0]
    subgrad = np.zeros_like(theta)
    # Accumulate subgradients for examples that violate margin
    margins = y * (Xb @ theta)
    violating = margins < 1
    if violating.any():
        subgrad = - (Xb[violating.flatten()].T @ y[violating]) / m
    # Add regularization term
    subgrad += lambda_ * theta
    return subgrad

def train_svm(Xb: np.ndarray, y: np.ndarray, lambda_: float = 0.01, lr: float = 0.01, epochs: int = 1000) -> tuple[np.ndarray, list]:
    """Train a linear soft‑margin SVM using subgradient descent."""
    theta = np.zeros((Xb.shape[1], 1))
    history = []
    for epoch in range(epochs):
        # Compute objective value (for monitoring)
        margins = y * (Xb @ theta)
        hinge_losses = np.maximum(0, 1 - margins)
        loss = hinge_losses.mean() + 0.5 * lambda_ * float((theta.T @ theta))
        history.append(loss)
        # Subgradient
        grad = svm_subgradient(Xb, y, theta, lambda_)
        theta -= lr * grad
    return theta, history


# %% [markdown]
# ### Training the SVM

lambda_ = 0.01
learning_rate = 0.01
epochs = 500

theta_svm, history_svm = train_svm(Xb, y, lambda_=lambda_, lr=learning_rate, epochs=epochs)

print(f"Final SVM objective value: {history_svm[-1]:.4f}")

# Plot objective over epochs
plt.figure(figsize=(6, 4))
plt.plot(history_svm)
plt.xlabel("Epoch")
plt.ylabel("Objective Value")
plt.title("Linear SVM Convergence")
plt.show()


# %% [markdown]
# ### Model Evaluation & Visualization

def predict_svm(Xb: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Predict class labels using the trained SVM."""
    return np.where((Xb @ theta) >= 0, 1, -1)

# Training accuracy
svm_preds = predict_svm(Xb, theta_svm)
svm_accuracy = (svm_preds == y).mean()
print(f"Training accuracy: {svm_accuracy * 100:.2f}%")

# Visualize decision boundary using two principal components
U, S, Vt = np.linalg.svd(X, full_matrices=False)
X_pc = X @ Vt.T[:, :2]
X_pc_b = np.hstack([np.ones((m, 1)), X_pc])
theta_pc, _ = train_svm(X_pc_b, y, lambda_=lambda_, lr=learning_rate, epochs=epochs)

# Compute line for decision boundary
theta0, theta1, theta2 = theta_pc.flatten()
x_vals = np.linspace(X_pc[:, 0].min() - 1, X_pc[:, 0].max() + 1, 200)
if abs(theta2) > 1e-6:
    y_vals = - (theta0 + theta1 * x_vals) / theta2
else:
    y_vals = np.zeros_like(x_vals)

# Plot
plt.figure(figsize=(6, 5))
plt.scatter(X_pc[y.flatten() == -1][:, 0], X_pc[y.flatten() == -1][:, 1], alpha=0.5, label="Class -1")
plt.scatter(X_pc[y.flatten() == 1][:, 0], X_pc[y.flatten() == 1][:, 1], alpha=0.5, label="Class +1")
plt.plot(x_vals, y_vals, 'r-', label="SVM decision boundary (PC space)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Linear SVM Decision Boundary")
plt.legend()
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Parameter Tuning**: Experiment with different values of \(\lambda\) and learning rate.  How does the
#    margin and training accuracy change?
# 2. **Pegasos Algorithm**: Implement the Pegasos stochastic subgradient method for SVMs.  Compare its
#    convergence to the batch subgradient descent used here.
# 3. **Nonlinear SVM**: Implement a kernelized SVM using the kernel trick (e.g., RBF kernel).  Use
#    scikit‑learn's SVM implementation for comparison, but derive the dual formulation yourself.
# 4. **Comparison to Logistic Regression**: Plot the decision boundary of a logistic regression classifier
#    trained on the same dataset in the PC space.  Discuss similarities and differences.
#
# ### Interview‑Ready Summary
#
# - Linear SVMs optimize a margin‑based objective: minimize hinge loss plus an
#   \(\ell_2\) regularization term.  The hinge loss penalizes misclassified and
#   insufficiently separated points.
# - The subgradient of the objective consists of a term involving only examples that violate
#   the margin and a regularization term proportional to \(\theta\).
# - Soft‑margin SVMs allow some misclassifications through the slack variables controlled
#   by the regularization parameter \(\lambda\).
# - Subgradient descent provides a simple way to train linear SVMs.  The Pegasos algorithm
#   further improves efficiency by using stochastic updates.
# - SVMs can be extended to nonlinear decision boundaries via kernel functions, leading to
#   nonlinear SVMs that implicitly operate in high‑dimensional feature spaces.
