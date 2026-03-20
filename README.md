# KNN Classification Project

## Overview
This project is a machine learning application developed in Python that performs classification using the K-Nearest Neighbors (KNN) algorithm.

The goal of the project is to demonstrate basic machine learning concepts such as data preprocessing, feature scaling, model training, and evaluation.

---

## Features
- Data loading and preprocessing
- Feature scaling using `StandardScaler`
- Train-test split with stratification
- KNN classification model
- Model evaluation using:
  - Accuracy score
  - Confusion matrix

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

---

## Dataset

The dataset file `baza_neutral.csv` is not included in this repository due to its size.

To run the project, please add the dataset file manually to the project folder.

### Requirements
- Place the file `baza_neutral.csv` in the root directory of the project.
- Make sure the file name matches exactly.

### Note
If the dataset is not found, the program will display an error message and cannot run properly.

To safely handle missing dataset, the following code block is used:

```python
try:
    data = pd.read_csv("baza_neutral.csv")
except:
    print("Dataset not found. Please add the dataset to run the code.")
```


## How to Run

### 1. Install dependencies
---
pip install -r requirements.txt

## Project Structure
- main.py
- requirements.txt
- README.md
- baza_neutral.csv (not included)

### 2. Run the program
python main.py


---

## Model Details
- Algorithm: K-Nearest Neighbors (KNN)
- Distance metric: Manhattan
- Number of neighbors: 10
- Data scaling: StandardScaler
- Train-test split: 80/20 with stratification

---

## Evaluation
The model is evaluated using:
- Accuracy score
- Confusion matrix

---

## Future Improvements
- Hyperparameter tuning (e.g. different values of `k`)
- Handling imbalanced datasets
- Trying other models (Logistic Regression, SVM)
- Adding data visualization

