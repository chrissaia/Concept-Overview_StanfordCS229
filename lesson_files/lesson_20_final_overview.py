# %% [markdown]
# # Lesson 20 — Course Review & Advanced Topics
#
# Congratulations on reaching the final lesson of *CS229 From Scratch*!  In this
# notebook we recap the key concepts covered throughout the course and briefly
# survey advanced topics in machine learning and reinforcement learning beyond
# the scope of this series.  Use this overview to consolidate your understanding
# and identify areas for further study.
#
# ## Recap of Core Concepts
#
# - **Supervised Learning**: regression (linear and logistic), margin‑based methods
#   (perceptron, SVM), generative models (GDA, Naive Bayes).
# - **Model Complexity & Overfitting**: bias–variance trade‑off, cross‑validation,
#   regularization, feature selection.
# - **Decision Trees & Ensembles**: tree learning with Gini impurity, bagging and
#   the foundation of random forests and boosting.
# - **Neural Networks**: feedforward architectures, backpropagation, loss functions,
#   optimization techniques.
# - **Unsupervised Learning**: k‑means clustering, Gaussian mixture models via
#   expectation–maximization, dimensionality reduction with PCA.
# - **Reinforcement Learning**: dynamic programming (value iteration), temporal
#   difference methods (TD, Q‑learning), policy gradients, linear quadratic
#   regulators.
#
# ## Advanced Topics & Next Steps
#
# - **Deep Learning**: convolutional and recurrent neural networks for image and
#   sequence data; transfer learning; self‑supervised learning.
# - **Probabilistic Graphical Models**: Bayesian networks and Markov random fields;
#   variational inference and sampling techniques.
# - **Kernel Methods**: kernelized SVMs and Gaussian processes for non‑linear
#   modelling without explicit feature mapping.
# - **Ensemble Methods**: boosting (e.g., AdaBoost, gradient boosting machines),
#   stacking and blending.
# - **Bayesian Optimization**: hyperparameter tuning using probabilistic surrogate
#   models and acquisition functions.
# - **Representation Learning**: autoencoders, variational autoencoders (VAEs) and
#   generative adversarial networks (GANs).
# - **Deep Reinforcement Learning**: policy gradients with function approximation
#   (A2C/A3C, PPO), Q‑learning with deep networks (DQN), model‑based RL.
# - **Meta‑Learning**: learn to learn; few‑shot learning and optimization‐based
#   meta‑learning (e.g., MAML).
#
# ## Further Reading
#
# - *Understanding Machine Learning: From Theory to Algorithms* by Shai Shalev‑Shwartz and Shai Ben‑David.
# - *Pattern Recognition and Machine Learning* by Christopher Bishop.
# - *Reinforcement Learning: An Introduction* by Sutton & Barto.
# - *Deep Learning* by Goodfellow, Bengio & Courville.
# - Lecture notes, problem sets and projects from the Stanford CS229 course.
#
# ## Closing Thoughts
#
# Building machine learning algorithms from scratch fosters a deeper appreciation
# of their assumptions, limitations and inner workings.  Use these notebooks as
# a springboard to tackle real‑world problems, experiment with new ideas and
# contribute to the ever‑evolving field of AI.


# %%
import numpy as np

np.random.seed(0)
