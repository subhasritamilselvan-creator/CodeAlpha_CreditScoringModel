# Credit Scoring Model

## 1. Project Overview

The Credit Scoring Model is a machine learning project that predicts whether a loan applicant may default on a loan.

The project uses Logistic Regression and Random Forest classification algorithms to make predictions and compare model performance.

## 2. Project Objectives

- Understand and preprocess credit risk data.
- Build machine learning classification models.
- Predict loan default status.
- Evaluate model performance.
- Compare Logistic Regression and Random Forest.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## 4. Dataset

The project uses a credit risk dataset containing applicant information such as:

- Age
- Income
- Home ownership
- Employment length
- Loan amount
- Loan interest rate
- Loan intent
- Credit history

The target variable is `loan_status`.

- 0 – No loan default
- 1 – Loan default

## 5. Data Preprocessing

The following preprocessing steps were performed:

- Loaded the dataset using Pandas.
- Checked and handled missing values.
- Removed invalid age records.
- Converted categorical variables into numerical values.
- Applied feature scaling to numerical columns.
- Split the dataset into training and testing sets.

## 6. Machine Learning Models

### Logistic Regression

Logistic Regression was trained to predict loan default status.

### Random Forest

Random Forest was trained using multiple decision trees to improve classification performance.

## 7. Model Performance

The observed accuracy results were:

| Model | Accuracy |
|---|---:|
| Logistic Regression | Approximately 87% |
| Random Forest | Approximately 93% |

Random Forest achieved higher accuracy in this experiment.

## 8. Visualizations

The project includes:

- Logistic Regression confusion matrix
- Model accuracy graph
- Classification metrics graph
- Random Forest confusion matrix
- Model accuracy comparison graph

All graphs are saved in the `graphs` folder.

## 9. Saved Models

- `credit_scoring_model.pkl`
- `random_forest_model.pkl`

The trained models were saved using Joblib.

## 10. Conclusion

This project demonstrates how machine learning algorithms can be used to study credit risk and predict loan default status.

Both models were evaluated, and Random Forest achieved higher accuracy in this experiment.

**Note:** This is an educational project. The models have not been validated for real-world lending decisions.