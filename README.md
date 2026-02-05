# Concept-Overview_Stanford CS229 (End-to-End Reconstruction)

This repository is a **ground-up reconstruction of Stanford CS229 (Machine Learning)** as a sequence of **Jupyter notebooks (Lesson 01 → Lesson 20)**.

This is **not** a summary repo. Each lesson is designed as a **teachable, executable chapter** with:

- deep intuition
- correct math (CS229-style derivations)
- minimal but correct NumPy implementations
- high-quality visualizations and diagnostics
- explicit connections across lessons

The sequence follows Stanford Engineering Everywhere’s CS229 lecture topics and handouts.

---

## Repository Philosophy

### 1) Implement the CS229 Version
If Andrew Ng derives something a certain way, this repo implements **that version** (or something similar), explains *why* it works, and sometimes where it fails.

### 2) NumPy First
We avoid library abstractions whenever possible.  
`sklearn` is used only for:
- loading datasets
- optional baseline comparisons

### 3) Endless Math
- MLE, objectives, gradients, etc.

### 4) Visualizations
Decision boundaries, probability surfaces, contours, margins, and diagnostics are part of the explanation.

---

## Quickstart

### Option A — Run Locally (Recommended)

```bash
git clone https://github.com/chrissaia/Concept-Overview_StanfordCS229
cd Concept-Overview_StanfordCS229

python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

pip install -U pip
pip install numpy matplotlib jupyter scipy pandas
```

Then:

```bash
jupyter lab
```

---

## Lesson Map (CS229-Aligned)

### Supervised Learning Core

| Lesson | Topic | Focus |
|------:|------|------|
| 01 | Introduction | No notebook needed |
| 02 | Linear Regression | Gradient descent, SGD, bias/variance |
| 03 | Logistic Regression + LWLR | Probabilistic view, Newton’s method |
| 04 | Generalized Linear Models | Exponential family, link functions |
| 05 | Generative Learning | GDA, Naive Bayes |
| 06 | Support Vector Machines | Margins, hinge loss |
| 07 | SVM Dual + KKT | Dual formulation |
| 08 | Kernels | Nonlinear decision boundaries |

---

### Unsupervised Learning

| Lesson | Topic | Focus |
|------:|------|------|
| 12 | K-Means + EM | Clustering |
| 13 | Gaussian Mixtures | EM |
| 16 | PCA + ICA | Dimensionality reduction |

---

### Reinforcement Learning

| Lesson | Topic | Focus |
|------:|------|------|
| 17 | MDPs | Value iteration |
| 18 | Continuous MDPs | Simulators |
| 19 | Control | LQR |
| 20 | Policy Search | REINFORCE |
