# CODEALPHA INTERNSHIP PROJECT REPORT

# CREDIT SCORING MODEL USING MACHINE LEARNING

## 1. Introduction

Credit scoring is an important application of machine learning in the financial sector. It helps analyse credit-related information and estimate the possibility of loan default.

This project develops a Credit Scoring Model using Python and machine learning classification algorithms.

## 2. Objective

The main objectives are:

- Analyse credit risk data.
- Clean and preprocess the dataset.
- Build classification models.
- Predict loan default status.
- Evaluate and compare model performance.

## 3. Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Visual Studio Code

## 4. Dataset Description

The dataset contains information about loan applicants, including:

- Person age
- Person income
- Home ownership
- Employment length
- Loan intent
- Loan grade
- Loan amount
- Loan interest rate
- Credit history

The target variable is `loan_status`.

- 0: No loan default
- 1: Loan default

After cleaning, the dataset contained approximately 32,411 records.

## 5. Data Preprocessing

The following steps were performed:

1. Loaded the dataset using Pandas.
2. Checked missing values.
3. Removed invalid age records.
4. Handled missing values.
5. Converted categorical features into numerical values.
6. Applied feature scaling.
7. Split the data into training and testing sets.

The dataset was split into 80% training data and 20% testing data.

## 6. Machine Learning Algorithms

### 6.1 Logistic Regression

Logistic Regression was implemented as the first classification algorithm.

It predicts the probability of a loan applicant belonging to a particular class.

Observed test accuracy: Approximately 87%.

### 6.2 Random Forest Classifier

Random Forest was implemented as the second classification algorithm.

It combines predictions from multiple decision trees to perform classification.

Observed test accuracy: Approximately 93%.

## 7. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Logistic Regression

Observed results:

- Accuracy: Approximately 87%
- Default-class precision: 0.78
- Default-class recall: 0.57
- Default-class F1-score: 0.66

### Random Forest

Observed results:

- Accuracy: Approximately 93%
- Default-class precision: 0.96
- Default-class recall: 0.72
- Default-class F1-score: 0.83

## 8. Model Comparison

| Model | Accuracy |
|---|---:|
| Logistic Regression | Approximately 87% |
| Random Forest | Approximately 93% |

Random Forest achieved higher test accuracy in this experiment.

## 9. Graphical Visualizations

The following graphs were generated and saved in the graphs folder:

1. Logistic Regression confusion matrix
2. Model accuracy graph
3. Classification metrics graph
4. Random Forest confusion matrix
5. Model accuracy comparison graph

## 10. Model Saving

The trained models were saved using Joblib.

- `credit_scoring_model.pkl`
- `random_forest_model.pkl`

This allows the saved models to be loaded for later prediction experiments.

## 11. Conclusion

The Credit Scoring Model project demonstrates the use of machine learning for analysing loan default risk.

Both Logistic Regression and Random Forest were implemented and evaluated.

Random Forest achieved approximately 93% accuracy, compared with approximately 87% for Logistic Regression.

Therefore, Random Forest performed better in this experiment based on accuracy and the observed default-class F1-score.

## 12. Limitations

- The models are educational prototypes.
- Accuracy alone does not establish model suitability.
- Further testing, validation and fairness analysis are needed.
- The models should not be used to make real lending decisions.

## 13. Project Structure

```text
CodeAlpha_CreditScoringModel/
│
├── graphs/
├── credit_risk_dataset.csv
├── credit_risk_cleaned.csv
├── credit_scoring.py
├── credit_scoring_model.pkl
├── random_forest_model.pkl
├── requirements.txt
├── README.md
└── Credit_Scoring_Model_Report.md
```