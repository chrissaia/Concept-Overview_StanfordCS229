# CS229 From Scratch

This repository turns each CS229 lecture topic into a self-contained Jupyter notebook with **CS229-style notation**, **from-scratch NumPy implementations**, **visualizations**, **exercises**, and **interview-style explanations**.

> **Note:** The official CS229 `main_notes.pdf` could not be fetched in this environment due to network restrictions. Each notebook includes a TODO reminder to validate equations once the PDF is available.

## How to run

1. Create a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open notebooks:
   ```bash
   jupyter lab
   ```

All notebooks are CPU-friendly and designed to run end-to-end in under ~5 minutes.

## Notes-driven syllabus (20 lessons)

Each lesson below maps to sections in the official CS229 lecture notes (main notes PDF) and is cross-checked against the local lecture transcripts in `transcripts/`.

| Lesson | Topic | Notes alignment (main notes sections) |
| --- | --- | --- |
| 01 | Intro, supervised learning, notation | Supervised learning overview, notation, linear regression setup |
| 02 | Linear regression (GD/SGD/normal equation) | Linear regression: cost, gradient descent, normal equation |
| 03 | Locally weighted regression + probabilistic view | Locally weighted regression; probabilistic interpretation of linear regression |
| 04 | Logistic regression + Newton + softmax + calibration | Logistic regression, Newton's method, softmax regression |
| 05 | Perceptron + exponential family + GLMs | Perceptron; exponential family; generalized linear models |
| 06 | Generative learning (GDA, LDA, QDA, Gaussian NB) | Generative learning algorithms, GDA, Naive Bayes |
| 07 | Naive Bayes for text + Laplace smoothing | Naive Bayes text classification, smoothing |
| 08 | Support vector machines | Margins, hinge loss, soft-margin SVM |
| 09 | Kernels | Kernel trick, representer theorem, kernelized methods |
| 10 | Bias/variance + regularization + model selection | Bias/variance, regularization, model selection |
| 11 | Learning theory | ERM, VC dimension, generalization bounds |
| 12 | Decision trees + ensembles | Decision trees, bagging, boosting (intro) |
| 13 | Neural networks (1–2 hidden layers, backprop) | Neural networks, backpropagation |
| 14 | k-means clustering | k-means clustering, diagnostics |
| 15 | PCA via SVD | PCA, dimensionality reduction |
| 16 | EM + Gaussian mixture models | EM algorithm, GMM |
| 17 | Anomaly detection | Gaussian anomaly detection |
| 18 | Collaborative filtering | Matrix factorization, collaborative filtering |
| 19 | Evaluation metrics | Confusion matrix, ROC/PR, thresholding |
| 20 | Reinforcement learning | MDPs, Bellman equations, value iteration |

## Progress table

| Lesson | Notebook | Status |
| --- | --- | --- |
| 01 | [lesson_01_intro_and_notation.ipynb](lesson_01_intro_and_notation.ipynb) | ✅ |
| 02 | [lesson_02_linear_regression.ipynb](lesson_02_linear_regression.ipynb) | ✅ |
| 02 (refined) | [lessons/lesson_02_linear_regression_refined.ipynb](lessons/lesson_02_linear_regression_refined.ipynb) | ✅ |
| 03 | [lesson_03_locally_weighted_regression.ipynb](lesson_03_locally_weighted_regression.ipynb) | ✅ |
| 04 | [lesson_04_logistic_regression_and_softmax.ipynb](lesson_04_logistic_regression_and_softmax.ipynb) | ✅ |
| 05 | [lesson_05_perceptron_and_glms.ipynb](lesson_05_perceptron_and_glms.ipynb) | ✅ |
| 06 | [lesson_06_generative_learning_gda.ipynb](lesson_06_generative_learning_gda.ipynb) | ✅ |
| 07 | [lesson_07_naive_bayes_text.ipynb](lesson_07_naive_bayes_text.ipynb) | ✅ |
| 08 | [lesson_08_svm.ipynb](lesson_08_svm.ipynb) | ✅ |
| 09 | [lesson_09_kernels.ipynb](lesson_09_kernels.ipynb) | ✅ |
| 10 | [lesson_10_bias_variance_regularization.ipynb](lesson_10_bias_variance_regularization.ipynb) | ✅ |
| 11 | [lesson_11_learning_theory.ipynb](lesson_11_learning_theory.ipynb) | ✅ |
| 12 | [lesson_12_decision_trees_and_ensembles.ipynb](lesson_12_decision_trees_and_ensembles.ipynb) | ✅ |
| 13 | [lesson_13_neural_networks.ipynb](lesson_13_neural_networks.ipynb) | ✅ |
| 14 | [lesson_14_k_means.ipynb](lesson_14_k_means.ipynb) | ✅ |
| 15 | [lesson_15_pca.ipynb](lesson_15_pca.ipynb) | ✅ |
| 16 | [lesson_16_em_gmm.ipynb](lesson_16_em_gmm.ipynb) | ✅ |
| 17 | [lesson_17_anomaly_detection.ipynb](lesson_17_anomaly_detection.ipynb) | ✅ |
| 18 | [lesson_18_collaborative_filtering.ipynb](lesson_18_collaborative_filtering.ipynb) | ✅ |
| 19 | [lesson_19_evaluation_metrics.ipynb](lesson_19_evaluation_metrics.ipynb) | ✅ |
| 20 | [lesson_20_reinforcement_learning.ipynb](lesson_20_reinforcement_learning.ipynb) | ✅ |

