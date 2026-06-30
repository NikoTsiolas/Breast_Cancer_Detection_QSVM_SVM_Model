# Breast Cancer Detection with SVM and QSVM

This project compares a classical Support Vector Machine model with a quantum Support Vector Machine approach for breast cancer classification.

The goal is simple: test whether a quantum machine learning approach can compete with a traditional SVM on a real breast cancer dataset, then compare the results in a practical way. I wanted to look beyond the hype and see how the models perform in terms of accuracy, recall, ROC AUC, and general usefulness.

This is not a medical diagnosis tool. It is a machine learning and quantum computing experiment.

## Overview

Breast cancer classification is a strong use case for testing machine learning models because the problem is clear: classify tumors as malignant or benign based on measured features.

This repository includes two main notebooks:

* `Breast_Cancer_SVM.ipynb`
* `Breast_Cancer_QSVM.ipynb`

The classical SVM notebook builds a traditional machine learning baseline using scikit-learn. The QSVM notebook explores the same type of classification problem through a quantum machine learning approach using Qiskit.

The point of the project is not to assume quantum is better. The point is to compare both approaches honestly and see where each one makes sense.

## Project Goals

* Build a classical SVM model for breast cancer classification
* Build a QSVM model for comparison
* Evaluate model performance using standard classification metrics
* Compare classical and quantum machine learning approaches
* Understand the practical tradeoffs between both methods

## Dataset

The project uses `Breast_cancer_data.csv`.

The dataset contains 569 rows and 6 columns:

* `mean_radius`
* `mean_texture`
* `mean_perimeter`
* `mean_area`
* `mean_smoothness`
* `diagnosis`

The target column is `diagnosis`, which represents whether the tumor is malignant or benign.

## Models

### Classical SVM

The classical model uses scikit-learn's SVM implementation. The notebook includes preprocessing, scaling, train-test splitting, hyperparameter tuning, feature selection, and model evaluation.

The SVM workflow includes:

* Loading the dataset
* Checking missing values and data types
* Splitting features and target labels
* Scaling the feature data
* Training an initial SVM model
* Running GridSearchCV for hyperparameter tuning
* Using RFECV for feature selection
* Evaluating model performance

### Quantum SVM

The QSVM notebook explores a quantum machine learning approach using Qiskit. The goal is to compare a quantum-based model against the classical SVM baseline and understand whether the quantum approach offers any practical advantage for this dataset.

## Results

The classical SVM performed well on the test set.

The hyperparameter-tuned SVM reached:

* Accuracy: 91.23%
* ROC AUC: 0.8948

The selected-feature SVM also reached:

* Accuracy: 91.23%
* ROC AUC: 0.8906

The original SVM baseline reached:

* Accuracy: 88.60%
* ROC AUC: 0.8692

These results show that tuning and feature selection improved the classical SVM compared to the original baseline.

The bigger takeaway is that classical machine learning is still very strong for structured datasets like this. Quantum machine learning is interesting and worth exploring, but it should be compared against strong classical baselines instead of being treated like an automatic upgrade.

## Technologies Used

* Python
* Jupyter Notebook
* pandas
* NumPy
* scikit-learn
* Qiskit
* Matplotlib
* Seaborn

## Repository Structure

```text
Breast_Cancer_Detection_QSVM_SVM_Model/
├── Breast_Cancer_QSVM.ipynb
├── Breast_Cancer_SVM.ipynb
├── Breast_cancer_data.csv
├── archive.zip
└── README.md
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/NikoTsiolas/Breast_Cancer_Detection_QSVM_SVM_Model.git
cd Breast_Cancer_Detection_QSVM_SVM_Model
```

Install the main dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn qiskit qiskit-machine-learning notebook
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open and run:

```text
Breast_Cancer_SVM.ipynb
Breast_Cancer_QSVM.ipynb
```

## What I Learned

This project helped me understand the difference between building a model that works and building a model that is actually useful.

The classical SVM was easier to train, easier to tune, and easier to evaluate. It also produced strong results on this dataset.

The QSVM approach was valuable because it showed how quantum machine learning can be applied to a real classification problem. At the same time, it made the limitations clear. Quantum models are not magic. They still need good preprocessing, clean data, careful feature selection, and fair comparison against classical models.

For this dataset, the classical SVM is the more practical model. The QSVM is useful as an experiment and as a way to better understand quantum machine learning.

## Future Improvements

Some improvements I would make next:

* Add a `requirements.txt` file for easier setup
* Clean up the notebook outputs
* Add a clearer side-by-side results table for SVM and QSVM
* Test more quantum feature maps
* Compare against additional classical models like Logistic Regression and Random Forest
* Add cross-validation results for both approaches
* Remove unnecessary files such as `.DS_Store`
* Add visualizations for confusion matrices and ROC curves

## Disclaimer

This project is for educational and experimental purposes only. It should not be used for medical diagnosis, clinical decisions, or treatment recommendations.
