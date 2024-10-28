# Fraud Transaction Detection Model
![image](https://github.com/user-attachments/assets/9628ff94-f5a3-4898-84e5-514da989f098)
![image](https://github.com/user-attachments/assets/ca3e9134-c0fb-459c-bef2-7c04c9ba1e9b)

This project demonstrates a machine learning approach to detecting potentially fraudulent credit card transactions. Using a dataset with imbalanced classes, various classification models such as Logistic Regression and Random Forest were trained, evaluated, and compared based on their precision, recall, F1 Score, and ROC AUC Score.

## Dataset

The data used in this project consists of anonymized credit card transactions, with features that capture various characteristics of each transaction. The dataset is highly imbalanced, containing only a small fraction of fraudulent transactions.

## Features

- **Time**: Time elapsed between this transaction and the first transaction in the dataset.
- **V1, V2, ..., V28**: Principal components obtained with PCA to protect the confidentiality of the data.
- **Amount**: Transaction amount.
- **Class**: Target variable (0 for legitimate transactions, 1 for fraudulent transactions).

## Project Workflow

1. **Data Preprocessing**: 
   - Handled missing values and scaled numeric fields.
   - Applied resampling techniques to balance the classes due to the significant class imbalance.

2. **Model Training and Evaluation**:
   - Trained two main classifiers: Logistic Regression and Random Forest.
   - Evaluated models based on Precision, Recall, F1 Score, and ROC AUC Score.

3. **Results**:
   - Compared model performances to determine the best classifier for detecting fraudulent transactions.

## Model Performance

The results showed the Random Forest classifier performed better than Logistic Regression, achieving higher precision and F1 Score while maintaining a good recall. This balance of precision and recall is crucial in fraud detection, where both false positives and false negatives have high costs.

## How to Use

1. Clone the repository.
2. Load the dataset (`creditcard.csv`).
3. Run `fraud_detection.py` to train the models and make predictions on new samples.

---

### Explanation of Code Section

#### Logistic Regression Training and Evaluation

```python
# Train and evaluate Logistic Regression
logistic_model.fit(X_train_res, y_train_res)
y_pred_logistic = logistic_model.predict(X_test)
```

- The code trains a `Logistic Regression` model on the resampled training data (`X_train_res`, `y_train_res`), which helps address the class imbalance in the dataset.
- The model makes predictions on the test data (`X_test`).

```python
print("Logistic Regression Performance:")
print("Precision:", precision_score(y_test, y_pred_logistic))
print("Recall:", recall_score(y_test, y_pred_logistic))
print("F1 Score:", f1_score(y_test, y_pred_logistic))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_logistic))
```

- **Precision**: Indicates the proportion of detected fraud cases that were actually fraudulent.
- **Recall**: Measures the model's ability to detect actual fraudulent cases.
- **F1 Score**: Balances precision and recall, which is crucial when the classes are imbalanced.
- **ROC AUC Score**: Reflects the model's ability to distinguish between classes across various thresholds, with higher values indicating better performance.

#### Random Forest Training and Evaluation

```python
# Train and evaluate Random Forest
random_forest_model.fit(X_train_res, y_train_res)
y_pred_rf = random_forest_model.predict(X_test)
```

- The `Random Forest` model is trained on the same resampled training data.
- Predictions are made on the test set.

```python
print("\nRandom Forest Performance:")
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_rf))
```

- Here, the `Random Forest` model outperforms Logistic Regression with a higher precision (88%), recall (80%), and F1 Score (84%).

#### Classification Report

```python
# Print classification report for more details
print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_logistic))
```

The classification report gives a detailed summary, including precision, recall, and F1-score for each class (0 for legitimate, 1 for fraud), overall accuracy, macro average, and weighted average, providing insight into the model's performance across different metrics. This is particularly helpful for understanding model performance on imbalanced classes.

