# heart_disease_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Heart_Disease_Prediction.csv")

# Data Preprocessing
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Convert categorical columns (like 'Presence' and 'Absence') to numerical values
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

# Feature selection
X = df.drop(columns=['Heart Disease'])  # Features
y = df['Heart Disease']  # Target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with Hyperparameter Tuning using GridSearchCV
log_reg_params = {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']}
log_reg = LogisticRegression()
log_reg_grid = GridSearchCV(log_reg, log_reg_params, cv=5, n_jobs=-1)
log_reg_grid.fit(X_train_scaled, y_train)

# Best Model from Logistic Regression
best_log_reg_model = log_reg_grid.best_estimator_
log_reg_y_pred = best_log_reg_model.predict(X_test_scaled)

# Logistic Regression Evaluation
log_reg_accuracy = accuracy_score(y_test, log_reg_y_pred)
print(f"Logistic Regression Accuracy: {log_reg_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, log_reg_y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, log_reg_y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Random Forest with Hyperparameter Tuning using GridSearchCV
rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
rf_model = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)

# Best Model from Random Forest
best_rf_model = rf_grid.best_estimator_
rf_y_pred = best_rf_model.predict(X_test_scaled)

# Random Forest Evaluation
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_y_pred))

# Cross-Validation Scores for Random Forest
cv_scores_rf = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=5)
print("\nRandom Forest Cross-Validation Scores:")
print(cv_scores_rf)
print(f"Average CV Score: {cv_scores_rf.mean():.4f}")
