pip install pandas numpy matplotlib seaborn scikit-learn

# Uber Fare Prediction using Machine Learning
# Models: Linear Regression and Random Forest Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import radians, sin, cos, sqrt, atan2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# ----------------------------------
# 1. LOAD DATASET FROM LOCAL CSV
# ----------------------------------

df = pd.read_csv("uber.csv")

print("First five rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ----------------------------------
# 2. DATA PREPROCESSING
# ----------------------------------

# Remove unnecessary columns
df.drop(columns=["key", "Unnamed: 0"],
        errors="ignore", inplace=True)

# Convert pickup_datetime into datetime format
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"],
    errors="coerce",
    utc=True
)

# Remove missing values
df.dropna(inplace=True)

# Extract date and time features
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year
df["weekday"] = df["pickup_datetime"].dt.dayofweek

# Remove original datetime column
df.drop(columns=["pickup_datetime"], inplace=True)

# Remove invalid fare values
df = df[(df["fare_amount"] > 0) &
        (df["fare_amount"] <= 500)]

# Remove invalid passenger counts
df = df[(df["passenger_count"] >= 1) &
        (df["passenger_count"] <= 6)]

# Validate geographical coordinates
df = df[
    df["pickup_latitude"].between(-90, 90) &
    df["dropoff_latitude"].between(-90, 90) &
    df["pickup_longitude"].between(-180, 180) &
    df["dropoff_longitude"].between(-180, 180)
]

# Remove coordinates outside the NYC study area
df = df[
    df["pickup_latitude"].between(40, 42) &
    df["dropoff_latitude"].between(40, 42) &
    df["pickup_longitude"].between(-75, -72) &
    df["dropoff_longitude"].between(-75, -72)
]

# ----------------------------------
# 3. CALCULATE DISTANCE
# ----------------------------------

def calculate_distance(lat1, lon1, lat2, lon2):

    # Convert degrees into radians
    lat1, lon1, lat2, lon2 = map(
        radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * \
        cos(lat2) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Earth radius in kilometres
    return 6371 * c


df["distance"] = df.apply(
    lambda row: calculate_distance(
        row["pickup_latitude"],
        row["pickup_longitude"],
        row["dropoff_latitude"],
        row["dropoff_longitude"]
    ),
    axis=1
)

# Remove zero and unusually large distances
df = df[
    (df["distance"] > 0) &
    (df["distance"] <= 100)
]

print("\nCleaned Dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)


# ----------------------------------
# 4. IDENTIFY OUTLIERS USING IQR
# ----------------------------------

print("\nOutlier Detection:")

for col in ["fare_amount", "distance", "passenger_count"]:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) | (df[col] > upper)
    ]

    print(col, "Outliers:", len(outliers))

# Remove outliers using IQR
for col in ["fare_amount", "distance"]:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[
        (df[col] >= lower) &
        (df[col] <= upper)
    ]


# ----------------------------------
# 5. CHECK CORRELATION
# ----------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()


# ----------------------------------
# 6. SELECT FEATURES AND TARGET
# ----------------------------------

X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------
# 7. LINEAR REGRESSION MODEL
# ----------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)

# Evaluation
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)

print("\nLINEAR REGRESSION RESULTS")
print("R2 Score:", lr_r2)
print("RMSE:", lr_rmse)
print("MAE:", lr_mae)


# ----------------------------------
# 8. RANDOM FOREST REGRESSION
# ----------------------------------

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# Evaluation
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)

print("\nRANDOM FOREST RESULTS")
print("R2 Score:", rf_r2)
print("RMSE:", rf_rmse)
print("MAE:", rf_mae)


# ----------------------------------
# 9. COMPARE BOTH MODELS
# ----------------------------------

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [lr_r2, rf_r2],
    "RMSE": [lr_rmse, rf_rmse],
    "MAE": [lr_mae, rf_mae]
})

print("\nMODEL COMPARISON:")
print(results)


# ----------------------------------
# 10. PLOT MODEL COMPARISON
# ----------------------------------

results.set_index("Model")[["R2 Score", "RMSE", "MAE"]].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Linear Regression vs Random Forest")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ----------------------------------
# 11. PREDICT UBER FARE
# ----------------------------------

print("\nUber Fare Prediction")

sample = X_test.iloc[[0]]

predicted_fare = rf.predict(sample)

print("Predicted Uber Fare:",
      round(predicted_fare[0], 2))
