# %% [markdown]
# # Lesson 11 — Practical Machine Learning Advice
#
# Machine learning practitioners often face challenges beyond implementing algorithms.
# Proper data splitting, feature scaling, regularization and debugging are crucial
# for building robust models.  In this notebook we illustrate some practical
# considerations using logistic regression on the breast cancer dataset.  We explore
# learning curves, the effect of regularization, and provide guidelines for
# diagnosing issues when a model underperforms.
#
# ## Outline
#
# - **Data splitting**: training, validation and test sets.
# - **Learning curves**: evaluate performance vs. training set size.
# - **Regularization**: observe impact of \(\ell_2\) regularization on logistic regression.
# - **Error analysis**: inspect misclassified examples.
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & Data Preparation

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets  # datasets only

np.random.seed(0)

# Load breast cancer dataset and standardize
X_raw, y_raw = datasets.load_breast_cancer(return_X_y=True)
X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
y = y_raw.reshape(-1, 1)

def split_train_val_test(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(X.shape[0])
    test_size = int(0.2 * X.shape[0])
    val_size = int(0.2 * X.shape[0])
    train_size = X.shape[0] - test_size - val_size
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    return X[train_idx], X[val_idx], X[test_idx], y[train_idx], y[val_idx], y[test_idx]

# Split into train (60%), validation (20%), test (20%)
X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

print(f"Training examples: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")


# %% [markdown]
# ### Logistic Regression with \(\ell_2\) Regularization
#
# We reuse the logistic regression functions from Lesson 3 but add an \(\ell_2\) penalty term.
# The regularized loss is
#
# \[
# J(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log (1 - h_\theta(x^{(i)}))] + \frac{\lambda}{2m} \lVert \theta\rVert^2.
# \]
#
# The gradient becomes
#
# \[
# \nabla J(\theta) = \frac{1}{m} X_b^T (h - y) + \frac{\lambda}{m} \theta.
# \]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss_reg(theta, Xb, y, lambda_):
    m = Xb.shape[0]
    h = sigmoid(Xb @ theta)
    h = np.clip(h, 1e-15, 1 - 1e-15)
    ce = - (y * np.log(h) + (1 - y) * np.log(1 - h)).mean()
    reg = (lambda_ / (2 * m)) * float(theta.T @ theta)
    return ce + reg

def grad_reg(theta, Xb, y, lambda_):
    m = Xb.shape[0]
    h = sigmoid(Xb @ theta)
    grad = (1 / m) * (Xb.T @ (h - y)) + (lambda_ / m) * theta
    return grad

def train_logistic_reg(Xb, y, lambda_=0.0, lr=0.1, epochs=1000):
    theta = np.zeros((Xb.shape[1], 1))
    for epoch in range(epochs):
        grad = grad_reg(theta, Xb, y, lambda_)
        theta -= lr * grad
    return theta

def predict(theta, Xb):
    return (sigmoid(Xb @ theta) >= 0.5).astype(int)


# Add intercept terms
X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_val_b = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])


# %% [markdown]
# ### Learning Curves
#
# We train logistic regression on increasing fractions of the training data and compute
# training and validation accuracies.  This helps diagnose high bias (both errors high and
# similar) vs. high variance (training error low, validation error high).

train_sizes = np.linspace(0.1, 1.0, 10)
train_accs = []
val_accs = []

lambda_reg = 0.0  # no regularization for learning curve

for frac in train_sizes:
    m_frac = int(frac * X_train.shape[0])
    idx = np.random.permutation(X_train.shape[0])[:m_frac]
    Xb_sub = X_train_b[idx]
    y_sub = y_train[idx]
    theta = train_logistic_reg(Xb_sub, y_sub, lambda_=lambda_reg, lr=0.1, epochs=500)
    train_pred = predict(theta, Xb_sub)
    val_pred = predict(theta, X_val_b)
    train_accs.append((train_pred == y_sub).mean())
    val_accs.append((val_pred == y_val).mean())

# Plot learning curves
plt.figure(figsize=(6, 4))
plt.plot(train_sizes * 100, np.array(train_accs) * 100, label="Training accuracy")
plt.plot(train_sizes * 100, np.array(val_accs) * 100, label="Validation accuracy")
plt.xlabel("Training set size (%)")
plt.ylabel("Accuracy (%)")
plt.title("Learning Curves (Logistic Regression)")
plt.legend()
plt.show()


# %% [markdown]
# ### Effect of Regularization
#
# We train logistic regression with different regularization strengths and evaluate on
# the validation set to see how \(\lambda\) affects performance.  Too little
# regularization may overfit, while too much causes underfitting.

lambda_values = [0.0, 0.01, 0.1, 1.0]
val_accs_reg = []
for lam in lambda_values:
    theta = train_logistic_reg(X_train_b, y_train, lambda_=lam, lr=0.1, epochs=800)
    val_pred = predict(theta, X_val_b)
    val_acc = (val_pred == y_val).mean()
    val_accs_reg.append(val_acc)
    print(f"Lambda={lam}, Validation accuracy: {val_acc * 100:.2f}%")

plt.figure(figsize=(6, 4))
plt.semilogx(lambda_values, np.array(val_accs_reg) * 100, marker='o')
plt.xlabel("Lambda (log scale)")
plt.ylabel("Validation Accuracy (%)")
plt.title("Effect of \u2113_2 Regularization on Accuracy")
plt.show()


# %% [markdown]
# ### Error Analysis
#
# Inspect misclassified examples on the validation set.  Error analysis can reveal
# systematic patterns or mislabeled data.  Here we simply count misclassifications.

theta_best = train_logistic_reg(X_train_b, y_train, lambda_=0.01, lr=0.1, epochs=800)
val_pred_best = predict(theta_best, X_val_b)
mis_idx = np.where(val_pred_best != y_val)[0]
print(f"Number of misclassified validation examples: {len(mis_idx)}")

# Print first few misclassified examples' indices and true/predicted labels
for i in mis_idx[:5]:
    print(f"Index {i}, True label: {int(y_val[i])}, Predicted: {int(val_pred_best[i])}")


# %% [markdown]
# ### Exercises
#
# 1. **Precision & Recall**: Compute precision, recall and F1 score for the logistic regression
#    classifier.  Discuss situations where accuracy is not sufficient.
# 2. **Feature Scaling**: Experiment with non‑standardized features.  How does lack of scaling
#    affect convergence and performance of gradient descent?
# 3. **Hyperparameter Tuning**: Implement grid search over learning rates and regularization
#    parameters using cross‑validation.  Plot validation accuracy as a function of hyperparameters.
# 4. **Debugging**: Intentionally break the gradient computation and observe how the
#    learning curves change.  Use gradient checking to detect bugs.
#
# ### Interview‑Ready Summary
#
# - Split data into training, validation and test sets to detect overfitting and tune
#   hyperparameters.  The test set should only be used once at the end of model selection.
# - Learning curves show how performance scales with more data; parallel training and
#   validation curves suggest high bias (both low) or high variance (training high, validation low).
# - Regularization prevents overfitting by penalizing large weights.  The regularization
#   strength \(\lambda\) must be tuned; too large causes underfitting.
# - Error analysis involves inspecting misclassified examples to identify patterns,
#   mislabeled data or feature inadequacies.
# - Proper feature scaling, debugging gradients and systematic hyperparameter tuning
#   are vital practical steps in building effective machine learning models.
