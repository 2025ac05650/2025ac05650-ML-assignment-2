# 2025ac05650-ML-assignment-2
ML Assignment 2 - Classification Models and Model Comparison


## Student Information

- **BITS ID:** 2025AC05650
- **NAME**: S Varun Reddy
- **Course:** Machine Learning
- **Assignment:** Assignment 2
  

---

# 1. Problem Statement

The objective of this assignment is to develop and compare multiple machine learning classification models for predicting whether a sales transaction will be returned.

The target variable is `Returned`:

- `0` - Not Returned
- `1` - Returned

Five different classification algorithms are implemented, trained, evaluated, and compared using multiple performance metrics.

---

# 2. Dataset Description

The dataset contains retail sales transaction information.

### Dataset Characteristics

- **Number of instances:** 1500
- **Original number of columns:** 19
- **Target variable:** `Returned`
- **Classification type:** Binary Classification

### Target Distribution

| Class | Description | Count |
|---|---|---:|
| 0 | Not Returned | 1128 |
| 1 | Returned | 372 |

The dataset is moderately imbalanced, with more transactions belonging to the Not Returned class.

---

# 3. Data Preprocessing

The following preprocessing steps were performed:

- Dataset loading and inspection
- Missing-value analysis
- Duplicate-value checking
- Removal of unnecessary columns
- Date feature extraction
- Numerical feature identification
- Categorical feature identification
- Missing-value handling
- Categorical feature encoding
- Numerical feature scaling
- Train-test splitting

### Feature Engineering

Date-related features were extracted from the date columns, including:

- Year
- Month
- Day
- Day of Week

After preprocessing and feature engineering, the model training pipeline used the resulting numerical representation of the features.

---

# 4. Train-Test Split

The original dataset was divided into training and testing sets.

| Dataset | Number of Samples |
|---|---:|
| Training | 1200 |
| Testing | 300 |
| Total | 1500 |

The test dataset contains:

- **226 Not Returned**
- **74 Returned**

The test data was kept separate for final model evaluation.

---

# 5. Machine Learning Models

Five classification algorithms were implemented.

## 5.1 Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the probability of a binary outcome.

## 5.2 Decision Tree

Decision Tree classification uses a sequence of feature-based decisions to divide the data into different classes.

## 5.3 K-Nearest Neighbors (KNN)

KNN classifies an observation based on the classes of its nearest neighboring observations.

## 5.4 Naive Bayes

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem and the assumption of conditional independence between features.

## 5.5 Random Forest

Random Forest is an ensemble learning method that combines predictions from multiple decision trees.

---

# 6. Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

These metrics provide different perspectives on model performance, particularly because the target classes are imbalanced.

---

# 7. Model Comparison

The models were evaluated on the 300-row test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7533 | 0.5179 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Decision Tree | 0.6600 | 0.4126 | 0.2407 | 0.1757 | 0.2031 | -0.0064 |
| KNN | 0.6867 | 0.4679 | 0.2222 | 0.1081 | 0.1455 | -0.0209 |
| Naive Bayes | 0.6933 | 0.4919 | 0.2632 | 0.1351 | 0.1786 | 0.0146 |
| Random Forest | 0.7567 | 0.4402 | 1.0000 | 0.0135 | 0.0267 | 0.1011 |

---

# 8. Model-wise Observations

## Logistic Regression

Logistic Regression achieved an accuracy of **75.33%** and an AUC of **0.5179**.

The model achieved relatively high accuracy because the dataset contains more Not Returned transactions. However, it failed to identify the Returned class effectively, with precision and recall of 0.00 for the positive class.

This indicates that accuracy alone is not sufficient to judge the performance of this model.

---

## Decision Tree

Decision Tree achieved an accuracy of **66.00%**.

It obtained:

- Precision: 0.2407
- Recall: 0.1757
- F1 Score: 0.2031
- MCC: -0.0064

The model showed a noticeable difference between training and testing performance, indicating possible overfitting.

---

## KNN

KNN achieved an accuracy of **68.67%**.

Its AUC was **0.4679**, while the precision and recall for the Returned class were relatively low.

The model did not perform strongly on identifying returned transactions.

---

## Naive Bayes

Naive Bayes achieved an accuracy of **69.33%**.

It achieved:

- Precision: 0.2632
- Recall: 0.1351
- F1 Score: 0.1786
- MCC: 0.0146

Naive Bayes provided moderate classification performance but was still limited in detecting the Returned class.

---

## Random Forest

Random Forest achieved the highest accuracy of **75.67%**.

It achieved:

- Precision: 1.0000
- Recall: 0.0135
- F1 Score: 0.0267
- MCC: 0.1011

Although the precision was 1.00, the very low recall indicates that the model detected almost none of the actual returned transactions.

Therefore, the high accuracy should be interpreted carefully.

---

# 9. Best Model

Based on the overall model comparison, **Random Forest** achieved the highest test accuracy:

**Accuracy = 75.67%**

It also achieved the highest MCC among the five models.

However, its recall for the Returned class was only **1.35%**.

Therefore, Random Forest achieved the best overall numerical performance according to the selected comparison criteria, but it is not highly effective at identifying actual returned transactions.

---

# 10. Important Observation

The dataset is imbalanced, with substantially more Not Returned transactions than Returned transactions.

Because of this imbalance, a model can achieve relatively high accuracy while performing poorly on the minority class.

For this problem, metrics such as:

- Recall
- F1 Score
- MCC
- AUC

should therefore be considered along with accuracy.

---

# 11. Project Files

The repository contains the following files:

```text
2025ac05650-ML-assignment-2/
│
├── README.md
├── app.py
├── requirements.txt
├── test_data.csv
│
├── logistic_regression.pkl
├── decision_tree.pkl
├── knn.pkl
├── naive_bayes.pkl
└── random_forest.pkl
