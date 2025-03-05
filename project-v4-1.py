import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("heart.csv")

# Data Preprocessing
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Fill missing values
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Convert categorical columns (like 'Presence' and 'Absence') to numerical values
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

# Feature selection
X = df.drop(columns=['target'])  # Features
y = df['target']  # Target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest with Hyperparameter Tuning using GridSearchCV
rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
rf_model = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)

# Best Model from Random Forest
best_rf_model = rf_grid.best_estimator_

# Get the predicted probabilities
rf_y_proba = best_rf_model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for class 1 (heart disease)

# Convert probabilities to percentage
rf_y_proba_percent = rf_y_proba * 100

# Show first few probabilities as percentages
for i in range(10):
    print(f"Sample {i+1}: {rf_y_proba_percent[i]:.2f}% chance of having heart disease")

# If you want to plot the probabilities as well
plt.figure(figsize=(10,6))
sns.histplot(rf_y_proba_percent, kde=True, color='blue')
plt.title('Probability Distribution of Heart Disease Prediction')
plt.xlabel('Probability of Heart Disease (%)')
plt.ylabel('Frequency')
plt.show()

# Random Forest Evaluation
rf_y_pred = best_rf_model.predict(X_test_scaled)

# Evaluate the model with accuracy and classification report
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Cross-Validation Scores for Random Forest
cv_scores_rf = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=5)
print("\nRandom Forest Cross-Validation Scores:")
print(cv_scores_rf)
print(f"Average CV Score: {cv_scores_rf.mean():.4f}")
