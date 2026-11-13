
# K-Nearest Neighbors on Diabetes Dataset
# SPPU Machine Learning Practical
# Python Only

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

# --------------------------------
# 1. LOAD DATASET
# --------------------------------

df = pd.read_csv("diabetes.csv")

print("First Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# --------------------------------
# 2. DATA PREPROCESSING
# --------------------------------

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Replace invalid zero values with median
# Zero is not a valid measurement for these columns
columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in columns:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# --------------------------------
# 3. SPLIT DATASET
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# --------------------------------
# 4. FEATURE SCALING
# --------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------
# 5. IMPLEMENT KNN ALGORITHM
# --------------------------------

knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Predict test data
y_pred = knn.predict(X_test)

# --------------------------------
# 6. CONFUSION MATRIX
# --------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)

plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("KNN Confusion Matrix")

plt.show()

# --------------------------------
# 7. ACCURACY
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

# --------------------------------
# 8. ERROR RATE
# --------------------------------

error_rate = 1 - accuracy

print("\nError Rate:", error_rate)
print("Error Percentage:", error_rate * 100, "%")

# --------------------------------
# 9. PRECISION
# --------------------------------

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nPrecision:", precision)

# --------------------------------
# 10. RECALL
# --------------------------------

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nRecall:", recall)

# --------------------------------
# 11. CLASSIFICATION REPORT
# --------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Diabetes", "Diabetes"],
        zero_division=0
    )
)

# --------------------------------
# 12. FINAL RESULT
# --------------------------------

print("\n----- FINAL RESULT -----")

print("Confusion Matrix:\n", cm)
print("Accuracy:", round(accuracy, 4))
print("Error Rate:", round(error_rate, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
