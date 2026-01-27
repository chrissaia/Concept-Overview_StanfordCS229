# %% [markdown]
# # Lesson 14 — Dimensionality Reduction: Principal Component Analysis
#
# Principal Component Analysis (PCA) is a widely used technique for reducing the
# dimensionality of a dataset while preserving as much variance as possible.  It
# projects data onto a lower‑dimensional subspace spanned by the leading eigenvectors
# of the covariance matrix.  In this notebook we implement PCA using the singular
# value decomposition (SVD) and apply it to the digits dataset.  We also explore
# reconstruction of compressed images.
#
# ## Outline
#
# - **Data loading & centering**: load digit images and subtract the mean.
# - **SVD & principal components**: compute components and explained variance.
# - **Projection & reconstruction**: compress data and reconstruct images.
# - **Visualization**: display original and reconstructed digits and plot variance ratios.
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & Data Preparation

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Load digits dataset (64 features, 8x8 images)
digits = datasets.load_digits()
X = digits.data  # shape (1797, 64)
images = digits.images

# Center the data (subtract mean)
X_mean = X.mean(axis=0)
X_centered = X - X_mean

print(f"Centered digits dataset: {X_centered.shape[0]} samples, {X_centered.shape[1]} features.")


# %% [markdown]
# ### PCA via Singular Value Decomposition

# Compute SVD of centered data matrix
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Explained variance of each component: (S^2)/(n_samples - 1)
explained_variances = (S ** 2) / (X_centered.shape[0] - 1)
explained_variance_ratio = explained_variances / explained_variances.sum()

# Plot cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA: Cumulative Explained Variance on Digits")
plt.grid(True)
plt.show()


# %% [markdown]
# ### Projection & Reconstruction
#
# We choose a small number of principal components (e.g. 16) and project the data
# onto this subspace.  We then reconstruct the images by projecting back to the
# original space and adding the mean.

n_components = 16
V_top = Vt[:n_components]  # shape (n_components, 64)

# Project data to lower dimension
X_projected = X_centered @ V_top.T
# Reconstruct from projection
X_reconstructed = X_projected @ V_top + X_mean

print(f"Data compressed from 64 to {n_components} dimensions.")

# Visualize original and reconstructed images for a few samples
num_samples = 5
indices = np.random.choice(X.shape[0], num_samples, replace=False)

plt.figure(figsize=(10, 4))
for i, idx in enumerate(indices):
    # Original
    ax = plt.subplot(2, num_samples, i + 1)
    ax.imshow(images[idx], cmap='gray')
    ax.axis('off')
    if i == 0:
        ax.set_title("Original")
    # Reconstructed
    ax = plt.subplot(2, num_samples, i + 1 + num_samples)
    ax.imshow(X_reconstructed[idx].reshape(8, 8), cmap='gray')
    ax.axis('off')
    if i == 0:
        ax.set_title("Reconstructed")
plt.suptitle(f"PCA Reconstruction with {n_components} Components")
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Choosing the Number of Components**: Find the minimal number of components that
#    explain at least 95% of the variance.  Visualize the cumulative variance curve
#    and justify your choice.
# 2. **Noise Reduction**: Add Gaussian noise to the digit images and perform PCA
#    reconstruction.  Observe how PCA can act as a denoising method by discarding
#    components corresponding to noise.
# 3. **Independent Component Analysis (ICA)**: Implement or use a library (e.g.,
#    scikit‑learn's `FastICA`) to extract independent components.  Compare the
#    components to the principal components.
# 4. **Applications**: Apply PCA to other datasets (e.g., face images, gene expression
#    data) and discuss its utility and limitations.
#
# ### Interview‑Ready Summary
#
# - **PCA** projects data onto orthogonal directions (principal components) that maximize
#   variance.  It can be computed via eigenvalue decomposition of the covariance
#   matrix or via singular value decomposition of the centered data matrix.
# - The explained variance ratio indicates how much information each principal
#   component retains; cumulative plots help decide how many components to keep.
# - PCA is commonly used for dimensionality reduction, visualization and noise
#   reduction.  Reconstruction error decreases as more components are retained.
# - Unlike PCA, **Independent Component Analysis (ICA)** seeks statistically independent
#   sources rather than uncorrelated directions.  ICA is useful for separating mixed
#   signals (e.g., cocktail party problem).
