# Data

This folder contains local datasets used by the notebooks. Some lessons rely on
scikit-learn's built-in toy datasets (downloaded automatically by `sklearn`),
while CSV files placed here are loaded directly from disk.

## Included files
- `energy_efficiency.csv`: optional regression dataset reserved for future or
  custom exercises. (Not referenced by the current lesson notebooks.)

## Dataset usage map

| Dataset | Source | Notebooks |
| --- | --- | --- |
| Diabetes | `sklearn.datasets.load_diabetes` | `lesson_02_linear_regression.ipynb` |
| Breast Cancer | `sklearn.datasets.load_breast_cancer` | `lesson_03_logistic_regression.ipynb`, `lesson_05_generative_models.ipynb`, `lesson_06_svm.ipynb`, `lesson_08_decision_trees.ipynb`, `lesson_11_practical_ml.ipynb` |
| Iris | `sklearn.datasets.load_iris` | `lesson_04_perceptron.ipynb` |
| Digits | `sklearn.datasets.load_digits` | `lesson_09_neural_networks.ipynb`, `lesson_14_pca.ipynb` |
| Synthetic blobs | `sklearn.datasets.make_blobs` | `lesson_12_kmeans.ipynb` |
| Synthetic mixtures | NumPy random draws | `lesson_13_em_gmm.ipynb` |

## Adding Kaggle datasets (manual download)

1. Visit the Kaggle dataset page and accept any license terms.
2. Click **Download** and save the archive locally.
3. Extract the CSV file into this `/data` folder.
4. Update this README with:
   - A short description of the dataset.
   - The Kaggle URL and any required citation/license notes.
   - The lesson notebook(s) that use the dataset and the expected filename.
5. In the notebook, load the CSV via a path like `data/<filename>.csv` and document
   any preprocessing steps in a markdown cell.

> Note: Notebooks should not use the Kaggle API or require authentication tokens.
