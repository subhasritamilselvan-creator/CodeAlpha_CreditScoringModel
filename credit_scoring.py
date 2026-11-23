
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv("credit_risk_dataset.csv")

# Display first five rows
print("Dataset loaded successfully!")

print(data.head())

# Display dataset size
print("\nDataset shape:")
print(data.shape)

# Display column names
print("\nColumn names:")
print(data.columns)


# DATA EXPLORATION

# 1. Display dataset information
print("\nDataset Information:")
print(data.info())

# 2. Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# 3. Check duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())

# 4. Display statistical information
print("\nStatistical Summary:")
print(data.describe())

# 5. Check loan status distribution
print("\nLoan Status Distribution:")
print(data["loan_status"].value_counts())

# DATA CLEANING

# 1. Count records before cleaning
print("\nBefore Cleaning:")
print(data.shape)

# 2. Remove duplicate records
data = data.drop_duplicates()

# 3. Count records after removing duplicates
print("\nAfter Removing Duplicates:")
print(data.shape)

# 4. Check unusual age records
print("\nRecords with age above 100:")
print(data[data["person_age"] > 100])

# 5. Save cleaned dataset
data.to_csv("credit_risk_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")


# CHECK UNUSUAL AGE RECORDS

print("\nUnusual Age Records:")

unusual_age = data[data["person_age"] > 100]

print(unusual_age)

print("\nNumber of unusual age records:")
print(unusual_age.shape[0])


# REMOVE INVALID AGE RECORDS

print("\nBefore Removing Invalid Ages:")
print(data.shape)

# Remove records where age is above 100
data = data[data["person_age"] <= 100].copy()

print("\nAfter Removing Invalid Ages:")
print(data.shape)

# Check again
print("\nRemaining Unusual Age Records:")
print(data[data["person_age"] > 100])

# Check missing values
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

# Save cleaned dataset
data.to_csv("credit_risk_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")


# STEP 3: SEPARATE FEATURES AND TARGET

from sklearn.model_selection import train_test_split

# Input features
X = data.drop("loan_status", axis=1)

# Target variable
y = data["loan_status"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# STEP 4: HANDLE MISSING VALUES

# Columns containing missing values
missing_columns = ["person_emp_length", "loan_int_rate"]

for column in missing_columns:

    # Calculate median from training data
    median_value = X_train[column].median()

    # Replace missing values in training data
    X_train[column] = X_train[column].fillna(median_value)

    # Replace missing values in testing data
    X_test[column] = X_test[column].fillna(median_value)

print("\nMissing Values After Imputation:")
print(X_train[missing_columns].isnull().sum())
print(X_test[missing_columns].isnull().sum())


# STEP 5: CATEGORICAL DATA ENCODING

# Convert categorical columns into numerical columns

X_train_encoded = pd.get_dummies(
    X_train,
    drop_first=True,
    dtype=int
)

X_test_encoded = pd.get_dummies(
    X_test,
    drop_first=True,
    dtype=int
)

# Match testing columns with training columns

X_test_encoded = X_test_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value=0
)

# Display encoded data

print("\nTraining Data After Encoding:")
print(X_train_encoded.head())

print("\nTesting Data After Encoding:")
print(X_test_encoded.head())

print("\nTraining Shape After Encoding:")
print(X_train_encoded.shape)

print("\nTesting Shape After Encoding:")
print(X_test_encoded.shape)


# STEP 6: FEATURE SCALING

from sklearn.preprocessing import StandardScaler

# Select numerical columns
numerical_columns = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

# Create StandardScaler object
scaler = StandardScaler()

# Copy encoded datasets
X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()

# Fit and transform training data
X_train_scaled[numerical_columns] = scaler.fit_transform(
    X_train_encoded[numerical_columns]
)

# Transform testing data
X_test_scaled[numerical_columns] = scaler.transform(
    X_test_encoded[numerical_columns]
)

# Display scaled data
print("\nTraining Data After Scaling:")
print(X_train_scaled.head())

print("\nTesting Data After Scaling:")
print(X_test_scaled.head())

print("\nTraining Shape After Scaling:")
print(X_train_scaled.shape)

print("\nTesting Shape After Scaling:")
print(X_test_scaled.shape)


# STEP 7: MODEL TRAINING

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_scaled, y_train)

# Display training message
print("\nModel Training Completed Successfully!")

# Check model information
print("\nModel:", model)

# Check training accuracy
print("\nTraining Accuracy:")
print(model.score(X_train_scaled, y_train))


# STEP 8: MODEL PREDICTION

# Predict loan status using testing data
y_pred = model.predict(X_test_scaled)

# Display first 10 predictions
print("\nFirst 10 Predictions:")
print(y_pred[:10])

# Display actual loan status
print("\nActual Loan Status:")
print(y_test.iloc[:10].values)

# Compare predicted and actual values
print("\nPrediction Completed Successfully!")


# STEP 9: MODEL EVALUATION

# Import evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# STEP 10: SAVE THE TRAINED MODEL

# Import joblib
import joblib

# Save model and preprocessing information
model_data = {
    "model": model,
    "scaler": scaler,
    "numerical_columns": numerical_columns,
    "feature_columns": X_train_encoded.columns.tolist()
}

# Save everything into a file
joblib.dump(model_data, "credit_scoring_model.pkl")

print("\nModel Saved Successfully!")
# STEP 11: VERIFY SAVED MODEL

import joblib

# Load saved model
saved_data = joblib.load("credit_scoring_model.pkl")

# Extract trained model
loaded_model = saved_data["model"]

# Make predictions using loaded model
test_predictions = loaded_model.predict(X_test_scaled)

print("\nSaved Model Loaded Successfully!")

print("\nFirst 10 Predictions Using Saved Model:")
print(test_predictions[:10])

print("\nModel Verification Completed!")


# STEP 12: MODEL VISUALIZATION

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Default", "Default"],
    yticklabels=["Not Default", "Default"]
)

plt.title("Credit Scoring - Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")

plt.show()


# Display model accuracy
plt.figure(figsize=(5, 4))

plt.bar(
    ["Logistic Regression"],
    [accuracy * 100]
)

plt.title("Model Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

plt.show()

print("\nVisualization Completed Successfully!")


# STEP 13: CLASSIFICATION METRICS VISUALIZATION

from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display values
print("\nClassification Metrics:")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Create graph
metrics = ["Precision", "Recall", "F1 Score"]
values = [precision, recall, f1]

plt.figure(figsize=(7, 5))

plt.bar(metrics, values)

plt.title("Classification Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)

plt.show()

print("\nClassification Metrics Graph Created!")


# STEP: RANDOM FOREST MODEL

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train_scaled, y_train)

print("\nRandom Forest Training Completed!")

# Make predictions
rf_predictions = rf_model.predict(X_test_scaled)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

# Classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))


from sklearn.metrics import accuracy_score, classification_report

# make random forest predictions
y_pred_rf = rf_model.predict(X_test_scaled)
# calculate accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Model Results")
print("---------------------------")
print("Accuracy:", rf_accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")

plt.savefig("graphs/random_forest_confusion_matrix.png")
plt.show()

import joblib

joblib.dump(rf_model, "random_forest_model.pkl")

print("Random Forest model saved successfully!")


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Calculate model accuracies
lr_accuracy = accuracy_score(y_test, y_pred)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# Model names
models = ["Logistic Regression", "Random Forest"]

# Accuracy values
accuracies = [
    lr_accuracy * 100,
    rf_accuracy * 100
]

# Create comparison graph
plt.figure(figsize=(8, 5))

plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

# Save graph
plt.savefig("graphs/model_comparison.png")

plt.show()
plt.close()

print("Model comparison graph saved successfully!")


import joblib

# Load Logistic Regression model
lr_model = joblib.load("credit_scoring_model.pkl")
print("Logistic Regression model loaded successfully!")

# Load Random Forest model
rf_model = joblib.load("random_forest_model.pkl")
print("Random Forest model loaded successfully!")
