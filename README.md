# CS229 From Scratch

This repository is a **from-scratch reconstruction of Stanford CS229** as a set of
self-contained Jupyter notebooks. Each lesson follows the CS229 lecture notes
structure and emphasizes:

- Clear notation and objective definitions.
- NumPy-only implementations (Matplotlib for plots; pandas only when loading CSVs).
- Visual intuition and diagnostic plots.
- Exercises and interview-style explanations.

Notebooks live in `lesson_files/` and are accompanied by their percent-format
Python scripts in the same directory.

## Project Philosophy

- **CS229-style notation**: align symbols, objectives, and derivations with the
  official lecture notes.
- **From-scratch NumPy implementations**: avoid model APIs and focus on the math.
- **Visualizations for intuition**: plots that make optimization and geometry concrete.
- **Exercises and reflections**: reinforce key ideas and common interview questions.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open any `lesson_files/lesson_XX_*.ipynb` notebook and run cells top-to-bottom.
All notebooks are CPU-friendly and designed to execute in under ~5 minutes.

## Progress

| Lesson | Topic | Notebook |
| --- | --- | --- |
| 02 | Linear regression (GD/SGD) | [lesson_02_linear_regression.ipynb](lesson_files/lesson_02_linear_regression.ipynb) |
| 03 | Logistic regression | [lesson_03_logistic_regression.ipynb](lesson_files/lesson_03_logistic_regression.ipynb) |
| 04 | Perceptron | [lesson_04_perceptron.ipynb](lesson_files/lesson_04_perceptron.ipynb) |
| 05 | Generative models (GDA + Naive Bayes) | [lesson_05_generative_models.ipynb](lesson_files/lesson_05_generative_models.ipynb) |
| 06 | Support Vector Machines | [lesson_06_svm.ipynb](lesson_files/lesson_06_svm.ipynb) |
| 07 | Bias/variance trade-off | [lesson_07_bias_variance.ipynb](lesson_files/lesson_07_bias_variance.ipynb) |
| 08 | Decision trees | [lesson_08_decision_trees.ipynb](lesson_files/lesson_08_decision_trees.ipynb) |
| 09 | Neural networks | [lesson_09_neural_networks.ipynb](lesson_files/lesson_09_neural_networks.ipynb) |
| 10 | Backpropagation | [lesson_10_backpropagation.ipynb](lesson_files/lesson_10_backpropagation.ipynb) |
| 11 | Practical ML | [lesson_11_practical_ml.ipynb](lesson_files/lesson_11_practical_ml.ipynb) |
| 12 | k-means clustering | [lesson_12_kmeans.ipynb](lesson_files/lesson_12_kmeans.ipynb) |
| 13 | EM for GMMs | [lesson_13_em_gmm.ipynb](lesson_files/lesson_13_em_gmm.ipynb) |
| 14 | PCA | [lesson_14_pca.ipynb](lesson_files/lesson_14_pca.ipynb) |
| 15 | Reinforcement learning: value iteration | [lesson_15_reinforcement_value_iteration.ipynb](lesson_files/lesson_15_reinforcement_value_iteration.ipynb) |
| 16 | Q-learning | [lesson_16_q_learning.ipynb](lesson_files/lesson_16_q_learning.ipynb) |
| 17 | TD learning | [lesson_17_td_learning.ipynb](lesson_files/lesson_17_td_learning.ipynb) |
| 18 | Policy gradients | [lesson_18_policy_gradient.ipynb](lesson_files/lesson_18_policy_gradient.ipynb) |
| 19 | LQR | [lesson_19_lqr.ipynb](lesson_files/lesson_19_lqr.ipynb) |
| 20 | Final overview | [lesson_20_final_overview.ipynb](lesson_files/lesson_20_final_overview.ipynb) |

> Lesson 01 will be added once the introductory linear regression notebook is
> migrated into the new `lesson_files/` structure.
