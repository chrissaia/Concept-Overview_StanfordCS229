# %% [markdown]
# # Lesson 8 — Decision Trees
#
# Decision trees recursively partition the feature space to form a tree of simple
# decision rules.  They are intuitive, handle nonlinear relationships and are the
# building blocks of powerful ensemble methods like random forests and boosting.
# In this notebook we implement a simple decision tree classifier from scratch
# using the Gini impurity criterion.
#
# ## Outline
#
# - **Data preparation**: load and preprocess a binary classification dataset.
# - **Impurity measures**: define Gini impurity and information gain.
# - **Tree building**: recursively split data to grow the tree.
# - **Prediction**: traverse the tree to classify new points.
# - **Visualization & evaluation**: plot decision boundaries for two features.
# - **Exercises & interview summary**.


# %% [markdown]
# ### Imports & Data Loading

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(0)

# Load the breast cancer dataset, restrict to two features for visualization
X_raw, y_raw = datasets.load_breast_cancer(return_X_y=True)
X = X_raw[:, :2]  # first two features
y = y_raw.reshape(-1, 1)

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

print(f"Dataset for decision tree: {X.shape[0]} examples, using 2 features.")


# %% [markdown]
# ### Gini Impurity and Best Split
#
# The Gini impurity of a binary node with class proportion \(p\) is \(2 p (1 - p)\).
# To find the best split along a feature, we consider threshold candidates between
# consecutive sorted values and compute the weighted impurity of the child nodes.

def gini_impurity(labels: np.ndarray) -> float:
    """Compute Gini impurity for binary labels (0/1)."""
    if len(labels) == 0:
        return 0.0
    p = labels.mean()
    return 2 * p * (1 - p)

def best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    """Find the best feature and threshold to split on using Gini impurity.

    Returns:
        best_feature_index, best_threshold, best_impurity
    """
    m, n = X.shape
    best_impurity = float('inf')
    best_feature = None
    best_thresh = None
    for j in range(n):
        # Sort examples by feature j
        sorted_idx = X[:, j].argsort()
        X_sorted = X[sorted_idx, j]
        y_sorted = y[sorted_idx]
        # Candidate thresholds: midpoints between unique values
        for i in range(1, m):
            if X_sorted[i] == X_sorted[i - 1]:
                continue
            thresh = 0.5 * (X_sorted[i] + X_sorted[i - 1])
            left_labels = y_sorted[:i]
            right_labels = y_sorted[i:]
            impurity = (len(left_labels) * gini_impurity(left_labels) + len(right_labels) * gini_impurity(right_labels)) / m
            if impurity < best_impurity:
                best_impurity = impurity
                best_feature = j
                best_thresh = thresh
    return best_feature, best_thresh, best_impurity


# %% [markdown]
# ### Decision Tree Classifier
#
# We build a binary tree recursively.  Each node stores the feature index and threshold for
# splitting, along with left and right child nodes.  Leaves store the predicted class.

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction  # class label for leaf nodes

def build_tree(X: np.ndarray, y: np.ndarray, depth: int = 0, max_depth: int = 3, min_samples_split: int = 2) -> TreeNode:
    """Recursively build a decision tree using Gini impurity."""
    # If all labels are the same or max depth reached, create a leaf
    if len(set(y.flatten())) == 1 or depth >= max_depth or len(y) < min_samples_split:
        # Predict majority class
        prediction = 1 if y.mean() >= 0.5 else 0
        return TreeNode(prediction=prediction)
    # Find best split
    feature, thresh, impurity = best_split(X, y)
    if feature is None:
        # Could not split (all feature values identical)
        prediction = 1 if y.mean() >= 0.5 else 0
        return TreeNode(prediction=prediction)
    # Partition data
    left_mask = X[:, feature] <= thresh
    right_mask = ~left_mask
    left_node = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples_split)
    right_node = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples_split)
    return TreeNode(feature=feature, threshold=thresh, left=left_node, right=right_node)

def tree_predict(x: np.ndarray, node: TreeNode) -> int:
    """Predict class label for a single example using the decision tree."""
    if node.prediction is not None:
        return node.prediction
    if x[node.feature] <= node.threshold:
        return tree_predict(x, node.left)
    else:
        return tree_predict(x, node.right)


# %% [markdown]
# ### Training the Tree and Evaluating

# Build a shallow tree for interpretability
tree_root = build_tree(X, y, max_depth=3)

# Predict on training data
y_pred = np.array([tree_predict(x, tree_root) for x in X]).reshape(-1, 1)
accuracy = (y_pred == y).mean()
print(f"Training accuracy of shallow decision tree: {accuracy * 100:.2f}%")


# %% [markdown]
# ### Visualization of Decision Boundaries
#
# We visualize the decision regions in the 2D feature space by evaluating the tree on a
# grid of points and plotting the resulting class predictions.

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

grid_pred = np.array([tree_predict(p, tree_root) for p in grid_points]).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, grid_pred, levels=[-0.5, 0.5, 1.5], alpha=0.3, cmap='bwr')
plt.scatter(X[y.flatten() == 0][:, 0], X[y.flatten() == 0][:, 1], label="Class 0", alpha=0.6)
plt.scatter(X[y.flatten() == 1][:, 0], X[y.flatten() == 1][:, 1], label="Class 1", alpha=0.6)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Tree Decision Regions (Depth 3)")
plt.legend()
plt.show()


# %% [markdown]
# ### Exercises
#
# 1. **Max Depth Tuning**: Train trees with different maximum depths and plot training vs. validation
#    accuracy to observe overfitting.
# 2. **Information Gain**: Implement a decision tree using information gain (entropy) instead of
#    Gini impurity and compare the resulting trees.
# 3. **Ensembles**: Implement bagging (bootstrap aggregating) by training multiple shallow trees
#    on bootstrap samples and averaging their predictions.  Evaluate the ensemble's accuracy.
# 4. **Continuous Splits**: Extend the tree to handle multi‑class problems and continuous splits
#    with more than two outcomes.
#
# ### Interview‑Ready Summary
#
# - Decision trees partition the feature space by recursively splitting on features that
#   maximize some purity criterion (e.g., Gini impurity or information gain).
# - A shallow tree may underfit, while a deep tree can overfit; controlling depth and
#   minimum samples per split mitigates overfitting.
# - Trees are interpretable: each path corresponds to a sequence of simple rules leading
#   to a prediction.
# - Ensembles like random forests and boosting use many trees to reduce variance and
#   improve performance over a single tree.
