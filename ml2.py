
# Email Spam Detection using KNN and SVM
# Python Only - SPPU Machine Learning Practical

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------
# 1. LOAD DATASET FROM LOCAL CSV
# ---------------------------------

df = pd.read_csv("emails.csv")

print("First Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# ---------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------

# Remove unnecessary Email No. column
df.drop(columns=["Email No."], inplace=True,
        errors="ignore")

# Remove missing values
df.dropna(inplace=True)

# Separate features and target
X = df.drop("Prediction", axis=1)
y = df["Prediction"]

print("\nClass Distribution:")
print(y.value_counts())

# ---------------------------------
# 3. SPLIT DATASET
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ---------------------------------
# 4. FEATURE SCALING
# ---------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------
# 5. K-NEAREST NEIGHBORS
# ---------------------------------

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

# KNN Evaluation
print("\n===== KNN CLASSIFICATION =====")

print("Accuracy:",
      accuracy_score(y_test, knn_pred))

print("Precision:",
      precision_score(y_test, knn_pred))

print("Recall:",
      recall_score(y_test, knn_pred))

print("F1 Score:",
      f1_score(y_test, knn_pred))

print("\nClassification Report:")
print(classification_report(y_test, knn_pred))

# KNN Confusion Matrix
cm_knn = confusion_matrix(y_test, knn_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_knn, annot=True, fmt="d",
            cmap="Blues")

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------------------
# 6. SUPPORT VECTOR MACHINE
# ---------------------------------

svm = SVC(kernel="linear")

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

# SVM Evaluation
print("\n===== SVM CLASSIFICATION =====")

print("Accuracy:",
      accuracy_score(y_test, svm_pred))

print("Precision:",
      precision_score(y_test, svm_pred))

print("Recall:",
      recall_score(y_test, svm_pred))

print("F1 Score:",
      f1_score(y_test, svm_pred))

print("\nClassification Report:")
print(classification_report(y_test, svm_pred))

# SVM Confusion Matrix
cm_svm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_svm, annot=True, fmt="d",
            cmap="Greens")

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------------------
# 7. PERFORMANCE COMPARISON
# ---------------------------------

results = pd.DataFrame({
    "Algorithm": ["KNN", "SVM"],

    "Accuracy": [
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, svm_pred)
    ],

    "Precision": [
        precision_score(y_test, knn_pred),
        precision_score(y_test, svm_pred)
    ],

    "Recall": [
        recall_score(y_test, knn_pred),
        recall_score(y_test, svm_pred)
    ],

    "F1 Score": [
        f1_score(y_test, knn_pred),
        f1_score(y_test, svm_pred)
    ]
})

print("\n===== MODEL COMPARISON =====")
print(results.to_string(index=False))

# ---------------------------------
# 8. COMPARISON GRAPH
# ---------------------------------

results.set_index("Algorithm").plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("KNN vs SVM Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ---------------------------------
# 9. FINAL RESULT
# ---------------------------------

if accuracy_score(y_test, knn_pred) > accuracy_score(
        y_test, svm_pred):
    print("\nKNN has better accuracy.")
elif accuracy_score(y_test, knn_pred) < accuracy_score(
        y_test, svm_pred):
    print("\nSVM has better accuracy.")
else:
    print("\nBoth models have equal accuracy.")
