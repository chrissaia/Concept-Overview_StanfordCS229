# %% [markdown]
# # Lesson 7 — Bias–Variance Trade‑off & Cross‑Validation
#
# Overfitting and underfitting are central themes in machine learning.  In this notebook
# we illustrate the **bias–variance trade‑off** by fitting polynomial regression models
# of varying degrees to noisy data.  We use cross‑validation to estimate the prediction
# error and discuss how model complexity affects bias and variance.
#
# ## Outline
#
# - **Synthetic data generation**: create a noisy nonlinear function.
# - **Polynomial feature construction**: map scalar inputs to polynomial basis.
# - **Model fitting**: fit models of different degrees via the normal equation.
# - **Training vs. validation error**: compute MSE on splits and plot.
# - **Visualization**: show model fits and error curves.
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & Data Generation

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Generate synthetic dataset
m = 200
X_raw = np.random.uniform(-2, 2, size=(m, 1))
# True function: sine wave
def f_true(x):
    return np.sin(np.pi * x)
y_true = f_true(X_raw)
# Add noise
noise = 0.3 * np.random.randn(m, 1)
y = y_true + noise

print(f"Generated {m} data points for bias–variance analysis.")


# %% [markdown]
# ### Polynomial Feature Construction

def poly_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Map input X (m x 1) to polynomial features up to the given degree."""
    X_poly = np.ones((X.shape[0], degree + 1))
    for d in range(1, degree + 1):
        X_poly[:, d] = X.flatten() ** d
    return X_poly

def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute closed‑form solution to linear regression via normal equations."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(((y_hat - y) ** 2).mean())


# %% [markdown]
# ### Cross‑Validation and Error Curves

# Split data into training (60%), validation (20%), and test (20%)
indices = np.random.permutation(m)
train_end = int(0.6 * m)
val_end = int(0.8 * m)
train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

X_train, y_train = X_raw[train_idx], y[train_idx]
X_val, y_val = X_raw[val_idx], y[val_idx]
X_test, y_test = X_raw[test_idx], y[test_idx]

# Fit models of increasing degree
degrees = list(range(0, 11))  # 0 to 10
train_errors = []
val_errors = []

for d in degrees:
    X_train_poly = poly_features(X_train, d)
    theta = normal_equation(X_train_poly, y_train)
    # Predictions
    train_pred = X_train_poly @ theta
    train_errors.append(mse(train_pred, y_train))
    # Validation
    X_val_poly = poly_features(X_val, d)
    val_pred = X_val_poly @ theta
    val_errors.append(mse(val_pred, y_val))

# Plot training and validation error vs. degree
plt.figure(figsize=(6, 4))
plt.plot(degrees, train_errors, label="Training error")
plt.plot(degrees, val_errors, label="Validation error")
plt.xlabel("Polynomial degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias–Variance Trade‑off via Polynomial Regression")
plt.legend()
plt.show()


# %% [markdown]
# ### Visualizing Model Fits
#
# We'll visualize the fitted curves for a low‑degree model (underfitting), an
# intermediate model (good fit) and a high‑degree model (overfitting) using the
# full dataset.

degrees_to_plot = [1, 3, 9]
plt.figure(figsize=(12, 4))
X_plot = np.linspace(-2, 2, 200).reshape(-1, 1)

for i, d in enumerate(degrees_to_plot):
    theta = normal_equation(poly_features(X_train, d), y_train)
    y_plot = poly_features(X_plot, d) @ theta
    plt.subplot(1, 3, i + 1)
    plt.scatter(X_train, y_train, alpha=0.3, label="Training data")
    plt.plot(X_plot, f_true(X_plot), 'g--', label="True function")
    plt.plot(X_plot, y_plot, 'r-', label=f"Degree {d} fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Model degree {d}")
    plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **K‑Fold Cross‑Validation**: Implement k‑fold cross‑validation (e.g., k=5) to select
#    the optimal polynomial degree based on validation error.  Compare to the simple
#    train/validation split used here.
# 2. **Regularization**: Fit polynomial regression with \(\ell_2\) regularization (ridge regression)
#    and explore how different regularization strengths affect the bias–variance trade‑off.
# 3. **Different True Functions**: Repeat the experiment with a different underlying
#    function (e.g., exponential, piecewise).  Observe how model complexity interacts
#    with the true function.
# 4. **Variance Decomposition**: Derive the bias and variance analytically for a simple
#    estimator (e.g., sample mean) and compare to empirical estimates.
#
# ### Interview‑Ready Summary
#
# - The **bias–variance trade‑off** captures the tension between underfitting and
#   overfitting: simple models have high bias but low variance, while complex models
#   have low bias but high variance.
# - Cross‑validation helps estimate prediction error by simulating unseen data.  The
#   validation error typically decreases then increases as model complexity grows,
#   illustrating the trade‑off.
# - Polynomial regression provides a simple playground for bias–variance experiments.
#   The normal equation gives the closed‑form solution, which can be regularized to
#   prevent overfitting.
# - Selecting model complexity based on validation error rather than training error
#   leads to better generalization.
