# %% [markdown]
# # Lesson 13 — Expectation–Maximization for Gaussian Mixture Models
#
# The Expectation–Maximization (EM) algorithm is a general technique for maximum
# likelihood estimation in latent variable models.  In this notebook we implement EM
# for a **Gaussian mixture model (GMM)**.  We generate synthetic data from a mixture
# of Gaussians, then use EM to recover the mixture parameters and visualize the
# clustering.
#
# ## Outline
#
# - **Data generation**: sample from a known mixture of Gaussians.
# - **Initialization**: guess initial component means, covariances and mixing weights.
# - **E‑step**: compute responsibilities (posterior probabilities of components).
# - **M‑step**: update parameters using responsibilities.
# - **Log‑likelihood monitoring**: track convergence.
# - **Visualization**: show estimated clusters and compare to ground truth.
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & Data Generation

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Generate synthetic data from 2 Gaussians
n_samples = 400
means_true = np.array([[0, 0], [5, 5]])
cov_true = np.array([[[1.0, 0.3], [0.3, 1.0]], [[1.5, -0.2], [-0.2, 1.0]]])
weights_true = np.array([0.4, 0.6])

components = np.random.choice(len(weights_true), size=n_samples, p=weights_true)
X = np.vstack([
    np.random.multivariate_normal(means_true[k], cov_true[k]) for k in components
])

print(f"Generated {n_samples} data points from a Gaussian mixture.")


# %% [markdown]
# ### EM Algorithm for GMM

def initialize_gmm(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize mixture weights, means and covariances."""
    n, d = X.shape
    weights = np.ones(k) / k
    # Randomly choose k data points as initial means
    indices = np.random.choice(n, k, replace=False)
    means = X[indices].copy()
    # Use identity matrices for initial covariances
    covariances = np.array([np.eye(d) for _ in range(k)])
    return weights, means, covariances

def multivariate_normal_pdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute multivariate normal PDF for each row in X."""
    d = mean.shape[0]
    cov = cov + 1e-6 * np.eye(d)
    cov_inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    norm_const = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
    diff = X - mean
    exponent = -0.5 * np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    return norm_const * np.exp(exponent)

def e_step(X: np.ndarray, weights: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """Expectation step: compute responsibilities."""
    n = X.shape[0]
    k = weights.shape[0]
    resp = np.zeros((n, k))
    for j in range(k):
        resp[:, j] = weights[j] * multivariate_normal_pdf(X, means[j], covs[j])
    # Normalize responsibilities
    resp_sum = resp.sum(axis=1, keepdims=True)
    resp = resp / resp_sum
    return resp

def m_step(X: np.ndarray, resp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maximization step: update weights, means and covariances."""
    n, d = X.shape
    k = resp.shape[1]
    nk = resp.sum(axis=0)  # effective number of points per component
    weights = nk / n
    means = (resp.T @ X) / nk[:, None]
    covs = np.zeros((k, d, d))
    for j in range(k):
        diff = X - means[j]
        covs[j] = (resp[:, j][:, None] * diff).T @ diff / nk[j]
        covs[j] += 1e-6 * np.eye(d)  # add small diagonal for stability
    return weights, means, covs

def log_likelihood(X: np.ndarray, weights: np.ndarray, means: np.ndarray, covs: np.ndarray) -> float:
    n = X.shape[0]
    k = weights.shape[0]
    ll = 0
    for i in range(n):
        prob = 0
        for j in range(k):
            prob += weights[j] * multivariate_normal_pdf(X[i:i+1], means[j], covs[j])[0]
        ll += np.log(prob + 1e-15)
    return ll

def em_gmm(X: np.ndarray, k: int, max_iters: int = 100, tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Run EM to fit a Gaussian mixture model."""
    weights, means, covs = initialize_gmm(X, k)
    ll_history = []
    for iteration in range(max_iters):
        # E‑step
        resp = e_step(X, weights, means, covs)
        # M‑step
        weights, means, covs = m_step(X, resp)
        # Log‑likelihood
        ll = log_likelihood(X, weights, means, covs)
        ll_history.append(ll)
        # Check convergence
        if iteration > 0 and abs(ll_history[-1] - ll_history[-2]) < tol:
            break
    return weights, means, covs, ll_history


# %% [markdown]
# ### Running EM and Visualizing

k = 2
weights_est, means_est, covs_est, ll_history = em_gmm(X, k)

print(f"EM converged in {len(ll_history)} iterations.")
print("Estimated weights:", weights_est)
print("Estimated means:\n", means_est)

# Plot log‑likelihood
plt.figure(figsize=(6, 4))
plt.plot(ll_history)
plt.xlabel("Iteration")
plt.ylabel("Log‑Likelihood")
plt.title("EM Log‑Likelihood for GMM")
plt.show()

# Plot data with estimated means and covariances
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Data")
for j in range(k):
    eigvals, eigvecs = np.linalg.eigh(covs_est[j])
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * np.sqrt(eigvals)
    ell = plt.matplotlib.patches.Ellipse(xy=means_est[j], width=width, height=height, angle=angle,
                                         edgecolor='red', fc='None', lw=2, label=f"Component {j}")
    plt.gca().add_patch(ell)
plt.scatter(means_est[:, 0], means_est[:, 1], color='red', marker='x', s=100, label="Estimated means")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Estimated Gaussian Components")
plt.legend()
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Number of Components**: Fit GMMs with different numbers of components and use the
#    Bayesian Information Criterion (BIC) to select k.
# 2. **Initialization Methods**: Try random initialization or k‑means based initialization
#    for EM.  Compare convergence speed and parameter estimates.
# 3. **Covariance Constraints**: Modify the algorithm to restrict covariances to be
#    diagonal (spherical) or tied across components.  Evaluate on high‑dimensional data.
# 4. **Soft Clustering**: Use the responsibilities to compute soft cluster memberships
#    and compare to hard assignments (argmax of responsibilities).
#
# ### Interview‑Ready Summary
#
# - The EM algorithm alternates between computing the expected latent variable
#   assignments (E‑step) and maximizing the expected complete‑data log‑likelihood with
#   respect to parameters (M‑step).  It monotonically increases the data likelihood.
# - In a Gaussian mixture model, the E‑step computes responsibilities proportional to
#   the weighted Gaussian densities.  The M‑step updates mixture weights, means and
#   covariances using these responsibilities.
# - Initialization matters; poor initial parameters can lead to local optima.  Multiple
#   restarts or k‑means initialization often improve results.
# - GMMs model arbitrary cluster shapes with full covariance matrices, unlike k‑means
#   which assumes spherical clusters.  Model selection criteria (e.g., BIC) help choose
#   the number of components.
