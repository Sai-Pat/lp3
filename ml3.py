pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

# Bank Customer Churn Prediction using ANN
# Artificial Neural Network - Binary Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------
# 1. READ DATASET
# ---------------------------------------

df = pd.read_csv("Churn_Modelling.csv")

print("First Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ---------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------

# Remove unnecessary columns
df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1,
    inplace=True,
    errors="ignore"
)

# Remove missing values
df.dropna(inplace=True)

# Separate features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=["Geography", "Gender"],
    drop_first=True,
    dtype=int
)

print("\nProcessed Features:")
print(X.head())

# ---------------------------------------
# 3. SPLIT DATASET
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ---------------------------------------
# 4. NORMALIZE DATA
# ---------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------
# 5. BASIC NEURAL NETWORK MODEL
# ---------------------------------------

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nBasic Neural Network:")
model.summary()

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Predict
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Basic model accuracy
basic_accuracy = accuracy_score(y_test, y_pred)

print("\nBasic ANN Accuracy:", basic_accuracy)

# ---------------------------------------
# 6. IMPROVED NEURAL NETWORK MODEL
# ---------------------------------------

improved_model = Sequential([
    Input(shape=(X_train.shape[1],)),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")
])

# Compile improved model
improved_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Calculate class weights
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\nImproved Neural Network:")
improved_model.summary()

# Train improved model
history = improved_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

# ---------------------------------------
# 7. PREDICTION
# ---------------------------------------

y_pred_prob = improved_model.predict(X_test)

y_pred = (y_pred_prob > 0.5).astype(int).ravel()

# ---------------------------------------
# 8. ACCURACY SCORE
# ---------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n===== IMPROVED ANN RESULTS =====")

print("Accuracy Score:", accuracy)

print("Accuracy Percentage:", accuracy * 100, "%")

# ---------------------------------------
# 9. CONFUSION MATRIX
# ---------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Exited", "Exited"],
    yticklabels=["Not Exited", "Exited"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Bank Customer Churn - Confusion Matrix")

plt.show()

# ---------------------------------------
# 10. CLASSIFICATION REPORT
# ---------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Exited", "Exited"]
    )
)

# ---------------------------------------
# 11. TRAINING ACCURACY GRAPH
# ---------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"],
         label="Training Accuracy")

plt.plot(history.history["val_accuracy"],
         label="Validation Accuracy")

plt.title("ANN Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.show()

# ---------------------------------------
# 12. FINAL RESULT
# ---------------------------------------

print("\n===== FINAL RESULT =====")

print("Basic ANN Accuracy:", basic_accuracy)

print("Improved ANN Accuracy:", accuracy)

if accuracy > basic_accuracy:
    print("Improved ANN performs better.")
elif accuracy < basic_accuracy:
    print("Basic ANN performs better.")
else:
    print("Both models have equal accuracy.")
