# %% [markdown]
# # Lesson 12 — Unsupervised Learning: k‑Means Clustering
#
# The k‑means algorithm partitions a set of data points into \(k\) clusters by iteratively
# assigning points to the nearest cluster centroid and updating centroids as the mean
# of assigned points.  It is a simple yet widely used unsupervised learning method.
# In this notebook we implement k‑means from scratch and apply it to synthetic data.
#
# ## Outline
#
# - **Data generation**: create synthetic clusters for visualization.
# - **k‑means algorithm**: initialization, assignment step, update step.
# - **Convergence criteria**: detect when cluster assignments stop changing.
# - **Visualization**: plot data colored by cluster and centroid trajectories.
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & Data Generation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs  # datasets only

np.random.seed(0)

# Generate synthetic data with 3 clusters
n_samples = 400
centers = [(-5, -2), (0, 0), (5, 5)]
cluster_std = [1.0, 1.5, 0.5]
X, y_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=42)

print(f"Generated {n_samples} data points for k‑means clustering.")


# %% [markdown]
# ### k‑Means Algorithm Implementation

def initialize_centroids(X: np.ndarray, k: int) -> np.ndarray:
    """Randomly select k data points as initial centroids."""
    indices = np.random.choice(X.shape[0], size=k, replace=False)
    return X[indices].copy()

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each point to the nearest centroid."""
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Compute new centroids as the mean of assigned points."""
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[labels == i]
        if len(points) > 0:
            centroids[i] = points.mean(axis=0)
    return centroids

def kmeans(X: np.ndarray, k: int, max_iters: int = 100) -> tuple[np.ndarray, np.ndarray, list]:
    """Run the k‑means clustering algorithm."""
    centroids = initialize_centroids(X, k)
    history = [centroids.copy()]
    for iteration in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        history.append(new_centroids.copy())
        # Check for convergence
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, labels, history


# %% [markdown]
# ### Running k‑Means and Visualizing

k = 3
centroids, labels, history = kmeans(X, k)

print(f"k‑Means converged in {len(history) - 1} iterations.")

# Plot final clustering
plt.figure(figsize=(6, 5))
for i in range(k):
    pts = X[labels == i]
    plt.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {i}", alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label="Centroids")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("k‑Means Clustering Results")
plt.legend()
plt.show()

# Plot centroid trajectory for each cluster
plt.figure(figsize=(6, 5))
for j in range(k):
    traj = np.array([c[j] for c in history])
    plt.plot(traj[:, 0], traj[:, 1], marker='o', label=f"Centroid {j} path")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Centroid Trajectories During k‑Means")
plt.legend()
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Initialization**: Experiment with different initialization strategies (e.g., k‑means++).
#    How does initialization affect convergence and cluster quality?
# 2. **Different k Values**: Vary k and use criteria like the elbow method to choose the
#    number of clusters.  Plot the within‑cluster sum of squares as a function of k.
# 3. **High‑Dimensional Data**: Apply k‑means to the digits dataset using PCA to reduce
#    dimensionality first.  Visualize the clusters in the reduced space.
# 4. **Spectral Clustering**: Implement spectral clustering and compare it to k‑means
#    on datasets with non‑convex clusters.
#
# ### Interview‑Ready Summary
#
# - The k‑means algorithm partitions data into k clusters by iteratively assigning
#   points to the nearest centroid and updating centroids to the mean of assigned
#   points.  It optimizes the sum of squared distances within clusters.
# - Convergence is guaranteed but the solution can depend on initialization; the
#   algorithm may converge to a local optimum.  k‑means++ is a common initialization
#   strategy that improves cluster quality.
# - Choosing the number of clusters k requires domain knowledge or heuristics such as
#   the elbow method or silhouette scores.
# - k‑means assumes spherical, equally sized clusters and uses Euclidean distance.
#   Variants like k‑medoids or Gaussian mixtures can handle different shapes and
#   distributions.
