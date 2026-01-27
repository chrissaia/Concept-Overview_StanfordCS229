# CS229 From Scratch

This repository is a **from-scratch reconstruction of Stanford CS229** in the form of self-contained Jupyter notebooks. Each lesson follows the CS229 lecture notes structure and focuses on:

- Clear notation and objective definitions.
- NumPy-only implementations (matplotlib for plots, pandas only when loading CSVs).
- Visual intuition and diagnostic plots.
- Exercises and interview-style explanations.

> **Primary sources**
> - Stanford CS229 lecture notes (main_notes.pdf).
> - Lecture transcripts in `transcripts/`.
> - The Lesson 02 template (refined copy included here).

## Notes-driven syllabus (20 lessons)

Each lesson maps directly to sections in the CS229 lecture notes PDF.

1. **Lesson 01** — Supervised learning, linear regression, least squares (Notes: Supervised Learning; Linear Regression).
2. **Lesson 02** — Gradient descent + stochastic gradient descent for linear regression (Notes: Linear Regression; Gradient Descent).
3. **Lesson 03** — Locally weighted regression + bias/variance intuition (Notes: Locally Weighted Regression; Bias/Variance).
4. **Lesson 04** — Logistic regression (Notes: Logistic Regression).
5. **Lesson 05** — Softmax regression + calibration (Notes: Multiclass Logistic Regression).
6. **Lesson 06** — Generalized linear models + link functions (Notes: GLMs).
7. **Lesson 07** — Generative learning + Gaussian Naive Bayes (Notes: Generative Learning Algorithms; Naive Bayes).
8. **Lesson 08** — LDA (Notes: Gaussian Discriminant Analysis).
9. **Lesson 09** — QDA (Notes: Gaussian Discriminant Analysis).
10. **Lesson 10** — Regularization (L1/L2) (Notes: Regularization and Model Selection).
11. **Lesson 11** — Linear SVM + hinge loss (Notes: Support Vector Machines).
12. **Lesson 12** — Kernels + kernelized demo (Notes: Kernels).
13. **Lesson 13** — K-means clustering (Notes: K-means).
14. **Lesson 14** — PCA via SVD (Notes: Principal Components Analysis).
15. **Lesson 15** — EM + Gaussian mixture models (Notes: EM and Mixture of Gaussians).
16. **Lesson 16** — Anomaly detection (Notes: Anomaly Detection).
17. **Lesson 17** — Collaborative filtering / matrix factorization (Notes: Recommender Systems).
18. **Lesson 18** — Neural networks + backprop (Notes: Neural Networks).
19. **Lesson 19** — Evaluation: ROC/PR, confusion matrix (Notes: Model Evaluation).
20. **Lesson 20** — Learning curves + model selection (Notes: Bias/Variance; Regularization and Model Selection).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open any `lesson_XX_*.ipynb` notebook and run cells top-to-bottom. All notebooks are CPU-friendly and designed to execute in under ~5 minutes.

## Progress

| Lesson | Topic | Notebook |
|---|---|---|
| 01 | Linear regression basics + normal equation | [lesson_01_intro_linear_regression_normal_equation.ipynb](lesson_01_intro_linear_regression_normal_equation.ipynb) |
| 02 | Linear regression GD/SGD (refined) | [lessons/lesson_02_linear_regression_refined.ipynb](lessons/lesson_02_linear_regression_refined.ipynb) |
| 03 | Locally weighted regression + bias/variance | [lesson_03_locally_weighted_regression_bias_variance.ipynb](lesson_03_locally_weighted_regression_bias_variance.ipynb) |
| 04 | Logistic regression (binary) | [lesson_04_logistic_regression_binary.ipynb](lesson_04_logistic_regression_binary.ipynb) |
| 05 | Softmax regression + calibration | [lesson_05_softmax_calibration.ipynb](lesson_05_softmax_calibration.ipynb) |
| 06 | Generalized linear models | [lesson_06_glms_link_functions.ipynb](lesson_06_glms_link_functions.ipynb) |
| 07 | Generative learning: Gaussian Naive Bayes | [lesson_07_gaussian_naive_bayes.ipynb](lesson_07_gaussian_naive_bayes.ipynb) |
| 08 | LDA | [lesson_08_lda.ipynb](lesson_08_lda.ipynb) |
| 09 | QDA | [lesson_09_qda.ipynb](lesson_09_qda.ipynb) |
| 10 | Regularization (L1/L2) | [lesson_10_regularization_l1_l2.ipynb](lesson_10_regularization_l1_l2.ipynb) |
| 11 | Linear SVM | [lesson_11_linear_svm.ipynb](lesson_11_linear_svm.ipynb) |
| 12 | Kernels (concept + demo) | [lesson_12_kernels_demo.ipynb](lesson_12_kernels_demo.ipynb) |
| 13 | K-means clustering | [lesson_13_kmeans_clustering.ipynb](lesson_13_kmeans_clustering.ipynb) |
| 14 | PCA via SVD | [lesson_14_pca_svd.ipynb](lesson_14_pca_svd.ipynb) |
| 15 | EM + Gaussian mixture models | [lesson_15_em_gmm.ipynb](lesson_15_em_gmm.ipynb) |
| 16 | Anomaly detection | [lesson_16_anomaly_detection.ipynb](lesson_16_anomaly_detection.ipynb) |
| 17 | Collaborative filtering | [lesson_17_collaborative_filtering_sgd.ipynb](lesson_17_collaborative_filtering_sgd.ipynb) |
| 18 | Neural networks (backprop) | [lesson_18_neural_network_backprop.ipynb](lesson_18_neural_network_backprop.ipynb) |
| 19 | Evaluation: ROC/PR, confusion matrix | [lesson_19_evaluation_metrics_roc_pr.ipynb](lesson_19_evaluation_metrics_roc_pr.ipynb) |
| 20 | Learning curves + model selection | [lesson_20_learning_curves_model_selection.ipynb](lesson_20_learning_curves_model_selection.ipynb) |
