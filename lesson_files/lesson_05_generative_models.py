# %% [markdown]
# # Lesson 5 — Generative Learning: GDA & Naive Bayes
#
# In this notebook we explore **generative models** for classification.  Instead of
# modeling \(P(y \mid x)\) directly as in discriminative methods like logistic regression,
# generative algorithms model the joint distribution \(P(x, y)\) or \(P(x \mid y) P(y)\) and then
# apply Bayes' rule.  We focus on two classical algorithms:
#
# 1. **Gaussian Discriminant Analysis (GDA)**: assumes the class‑conditional distribution
#    \(x \mid y\) is multivariate normal with class‑specific means and a shared covariance
#    matrix.
# 2. **Naive Bayes**: assumes features are conditionally independent given the class.
#    We implement a Gaussian naive Bayes classifier where each feature is modeled as
#    univariate normal.
#
# ## Outline
#
# - **Data loading & preprocessing**: binary classification dataset.
# - **GDA derivation & implementation**: estimate parameters and predict.
# - **Gaussian Naive Bayes**: estimate means/variances for each feature per class.
# - **Comparison & evaluation**: accuracy and decision boundaries.
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & Data Preparation

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(0)

# Load breast cancer dataset
X_raw, y_raw = datasets.load_breast_cancer(return_X_y=True)

# Standardize features
X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
y = y_raw.reshape(-1, 1)

# Split into training and testing sets (80/20 split)
def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    indices = rng.permutation(m)
    test_size = int(m * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

X_train, y_train, X_test, y_test = train_test_split(X, y)
m_train, n = X_train.shape

print(f"Training set: {m_train} examples, Testing set: {X_test.shape[0]} examples")


# %% [markdown]
# ### Gaussian Discriminant Analysis (GDA)
#
# In GDA we assume that for each class \(c \in \{0,1\}\), the conditional distribution
# \(x \mid y=c\) is multivariate normal with mean \(\mu_c\) and shared covariance matrix
# \(\Sigma\).  The class priors are \(\phi_c = P(y=c)\).  The parameters are estimated as:
#
# \[
# \phi_c = \frac{1}{m} \sum_{i=1}^m 1\{y^{(i)} = c\},\quad
# \mu_c = \frac{\sum_{i=1}^m 1\{y^{(i)} = c\} x^{(i)}}{\sum_{i=1}^m 1\{y^{(i)} = c\}},\quad
# \Sigma = \frac{1}{m} \sum_{i=1}^m \bigl(x^{(i)} - \mu_{y^{(i)}}\bigr) \bigl(x^{(i)} - \mu_{y^{(i)}}\bigr)^T.
# \]
#
# The log posterior difference for class 1 vs class 0 can be written as a linear function of
# \(x\).  We derive a weight vector and bias term that yields a logistic form similar to
# logistic regression.

def gda_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate parameters for Gaussian Discriminant Analysis.

    Returns:
        phi (np.ndarray): class prior for y=1
        mu_0, mu_1 (np.ndarray): class means (shape (n, 1))
        Sigma (np.ndarray): shared covariance matrix (shape (n, n))
    """
    m, n = X.shape
    y = y.flatten()
    phi = y.mean()
    mu_0 = X[y == 0].mean(axis=0).reshape(-1, 1)
    mu_1 = X[y == 1].mean(axis=0).reshape(-1, 1)
    Sigma = np.zeros((n, n))
    for i in range(m):
        xi = X[i].reshape(-1, 1)
        mu = mu_1 if y[i] == 1 else mu_0
        Sigma += (xi - mu) @ (xi - mu).T
    Sigma /= m
    return phi, mu_0, mu_1, Sigma

def gda_predict(X: np.ndarray, phi: float, mu_0: np.ndarray, mu_1: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Predict class labels using GDA parameters."""
    # Compute inverse and determinant once
    Sigma_inv = np.linalg.pinv(Sigma)
    # Discriminant scores for class 1 and 0
    def log_gaussian(x: np.ndarray, mu: np.ndarray) -> float:
        diff = x - mu
        return -0.5 * float(diff.T @ Sigma_inv @ diff)
    preds = []
    for xi in X:
        xi_col = xi.reshape(-1, 1)
        log_p1 = log_gaussian(xi_col, mu_1) + np.log(phi + 1e-15)
        log_p0 = log_gaussian(xi_col, mu_0) + np.log(1 - phi + 1e-15)
        preds.append(1 if log_p1 > log_p0 else 0)
    return np.array(preds).reshape(-1, 1)


# Fit GDA parameters
phi, mu_0, mu_1, Sigma = gda_fit(X_train, y_train)

# Predict on test set and compute accuracy
gda_preds = gda_predict(X_test, phi, mu_0, mu_1, Sigma)
gda_accuracy = (gda_preds == y_test).mean()
print(f"GDA test accuracy: {gda_accuracy * 100:.2f}%")


# %% [markdown]
# ### Gaussian Naive Bayes
#
# Under the naive Bayes assumption, features are conditionally independent given the class,
# so the class‑conditional density factors as a product of univariate normals.  For each
# class \(c\) and feature \(j\), we estimate the mean \(\mu_{c,j}\) and variance \(\sigma^2_{c,j}\) from
# the training data.  The posterior can then be computed via
#
# \[
# P(y=1 \mid x) \propto \phi \prod_{j=1}^n \mathcal{N}(x_j; \mu_{1,j}, \sigma^2_{1,j}),
# \]
# and similarly for \(y=0\).  Taking logs avoids underflow.

def naive_bayes_fit(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate parameters for Gaussian Naive Bayes.

    Returns:
        phi: class prior for y=1
        mu_0, mu_1: mean vectors for each class (shape (n, 1))
        var_0, var_1: variance vectors for each class (shape (n, 1))
    """
    m, n = X.shape
    y = y.flatten()
    phi = y.mean()
    mu_0 = X[y == 0].mean(axis=0).reshape(-1, 1)
    mu_1 = X[y == 1].mean(axis=0).reshape(-1, 1)
    var_0 = X[y == 0].var(axis=0).reshape(-1, 1) + 1e-9  # add epsilon to avoid zeros
    var_1 = X[y == 1].var(axis=0).reshape(-1, 1) + 1e-9
    return phi, mu_0, mu_1, var_0, var_1

def naive_bayes_predict(X: np.ndarray, phi: float, mu_0: np.ndarray, mu_1: np.ndarray, var_0: np.ndarray, var_1: np.ndarray) -> np.ndarray:
    """Predict class labels using Gaussian Naive Bayes parameters."""
    n = X.shape[1]
    log_phi1 = np.log(phi + 1e-15)
    log_phi0 = np.log(1 - phi + 1e-15)
    preds = []
    for xi in X:
        xi_col = xi.reshape(-1, 1)
        # log likelihood for class 1
        log_likelihood1 = -0.5 * np.sum(np.log(2 * np.pi * var_1) + ((xi_col - mu_1) ** 2) / var_1)
        # log likelihood for class 0
        log_likelihood0 = -0.5 * np.sum(np.log(2 * np.pi * var_0) + ((xi_col - mu_0) ** 2) / var_0)
        log_p1 = log_phi1 + log_likelihood1
        log_p0 = log_phi0 + log_likelihood0
        preds.append(1 if log_p1 > log_p0 else 0)
    return np.array(preds).reshape(-1, 1)


# Fit Gaussian Naive Bayes parameters
phi_nb, mu_0_nb, mu_1_nb, var_0_nb, var_1_nb = naive_bayes_fit(X_train, y_train)

# Predict and evaluate
nb_preds = naive_bayes_predict(X_test, phi_nb, mu_0_nb, mu_1_nb, var_0_nb, var_1_nb)
nb_accuracy = (nb_preds == y_test).mean()
print(f"Gaussian Naive Bayes test accuracy: {nb_accuracy * 100:.2f}%")


# %% [markdown]
# ### Comparison & Visualization
#
# For a low‑dimensional visualization, we project the data onto the first two principal
# components using singular value decomposition.  We then plot the decision boundaries
# of GDA and Naive Bayes in this 2D space.  Note that the actual classification uses
# all features; the projection is for illustration only.

# Compute first two principal components
U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
X_train_pc = X_train @ Vt.T[:, :2]
X_test_pc = X_test @ Vt.T[:, :2]

# Fit models in PC space using only two features
phi_pc, mu0_pc, mu1_pc, Sigma_pc = gda_fit(X_train_pc, y_train)
phi_nb_pc, mu0_nb_pc, mu1_nb_pc, var0_nb_pc, var1_nb_pc = naive_bayes_fit(X_train_pc, y_train)

# Create grid for decision boundary visualization
x_min, x_max = X_train_pc[:, 0].min() - 1, X_train_pc[:, 0].max() + 1
y_min, y_max = X_train_pc[:, 1].min() - 1, X_train_pc[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

gda_grid_pred = gda_predict(grid, phi_pc, mu0_pc, mu1_pc, Sigma_pc).reshape(xx.shape)
nb_grid_pred = naive_bayes_predict(grid, phi_nb_pc, mu0_nb_pc, mu1_nb_pc, var0_nb_pc, var1_nb_pc).reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(xx, yy, gda_grid_pred, levels=[-0.5, 0.5, 1.5], alpha=0.3, cmap='bwr')
plt.scatter(X_train_pc[y_train[:, 0] == 0][:, 0], X_train_pc[y_train[:, 0] == 0][:, 1], label="Class 0", alpha=0.6)
plt.scatter(X_train_pc[y_train[:, 0] == 1][:, 0], X_train_pc[y_train[:, 0] == 1][:, 1], label="Class 1", alpha=0.6)
plt.title("GDA Decision Boundary (2D PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, nb_grid_pred, levels=[-0.5, 0.5, 1.5], alpha=0.3, cmap='bwr')
plt.scatter(X_train_pc[y_train[:, 0] == 0][:, 0], X_train_pc[y_train[:, 0] == 0][:, 1], label="Class 0", alpha=0.6)
plt.scatter(X_train_pc[y_train[:, 0] == 1][:, 0], X_train_pc[y_train[:, 0] == 1][:, 1], label="Class 1", alpha=0.6)
plt.title("Naive Bayes Decision Boundary (2D PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Class Imbalance**: Modify the class prior \(\phi\) artificially (e.g., set \(\phi = 0.9\))
#    and observe how the decision boundary and accuracy change.
# 2. **Multivariate vs. Naive**: Create a synthetic dataset where features are highly correlated.
#    Compare GDA and Naive Bayes performance.
# 3. **Categorical Naive Bayes**: Implement a Bernoulli or multinomial naive Bayes classifier for text
#    classification.  Apply it to a toy document classification problem.
# 4. **Regularized Covariance**: In GDA, add a small multiple of the identity matrix to \(\Sigma\)
#    to improve numerical stability.  Investigate its impact on performance.
#
# ### Interview‑Ready Summary
#
# - **Generative vs. Discriminative**: generative models learn \(P(x \mid y) P(y)\) and use Bayes' rule
#   to predict; discriminative models learn \(P(y \mid x)\) directly.
# - **Gaussian Discriminant Analysis** assumes a multivariate normal distribution for each class
#   with a shared covariance matrix.  The resulting decision boundary is linear and
#   resembles logistic regression.
# - **Naive Bayes** assumes conditional independence of features given the class.  The
#   generative distribution factorizes, simplifying parameter estimation but potentially
#   reducing accuracy when features are correlated.
# - Both GDA and Naive Bayes require estimating class priors and means.  Naive Bayes also
#   estimates variances per feature.  Adding Laplace smoothing or small variances
#   prevents numerical issues.
# - Generative models can be extended to multimodal distributions (e.g., Gaussian mixtures)
#   or other exponential family distributions depending on the data type.
