
# K-Means and Hierarchical Clustering
# SPPU Machine Learning Practical
# Dataset: sales_data_sample.csv
# Python Only

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage

# ---------------------------------------
# 1. LOAD DATASET
# ---------------------------------------

df = pd.read_csv(
    "sales_data_sample.csv",
    encoding="latin1"
)

print("First Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ---------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------

# Select important numerical features
data = df[
    ["QUANTITYORDERED", "PRICEEACH", "SALES"]
].copy()

# Handle missing values
data = data.fillna(data.median())

print("\nSelected Features:")
print(data.head())

# ---------------------------------------
# 3. FEATURE SCALING
# ---------------------------------------

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

print("\nScaled Data:")
print(scaled_data[:5])

# ---------------------------------------
# 4. ELBOW METHOD
# ---------------------------------------

wcss = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(scaled_data)

    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker="o",
    color="blue"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.xticks(range(1, 11))
plt.grid()
plt.show()

# ---------------------------------------
# 5. APPLY K-MEANS CLUSTERING
# ---------------------------------------

# Choose K by observing the elbow graph
k = 3

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["KMeans_Cluster"] = kmeans.fit_predict(
    scaled_data
)

print("\nK-Means Cluster Results:")
print(df[["SALES", "KMeans_Cluster"]].head())

print("\nK-Means Cluster Distribution:")
print(df["KMeans_Cluster"].value_counts())

# Visualize K-Means Clusters
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=data["QUANTITYORDERED"],
    y=data["SALES"],
    hue=df["KMeans_Cluster"],
    palette="viridis"
)

plt.title("K-Means Clustering")
plt.xlabel("Quantity Ordered")
plt.ylabel("Sales")
plt.show()

# ---------------------------------------
# 6. HIERARCHICAL CLUSTERING
# ---------------------------------------

hierarchical = AgglomerativeClustering(
    n_clusters=k,
    linkage="ward"
)

df["Hierarchical_Cluster"] = hierarchical.fit_predict(
    scaled_data
)

print("\nHierarchical Clustering Results:")
print(df[["SALES", "Hierarchical_Cluster"]].head())

print("\nHierarchical Cluster Distribution:")
print(df["Hierarchical_Cluster"].value_counts())

# ---------------------------------------
# 7. DENDROGRAM
# ---------------------------------------

# Use a sample to keep the dendrogram readable
sample_size = min(200, len(scaled_data))

sample_data = scaled_data[:sample_size]

linked = linkage(
    sample_data,
    method="ward"
)

plt.figure(figsize=(12, 6))

dendrogram(
    linked,
    truncate_mode="lastp",
    p=30
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points / Merged Clusters")
plt.ylabel("Euclidean Distance")
plt.show()

# ---------------------------------------
# 8. VISUALIZE HIERARCHICAL CLUSTERS
# ---------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=data["QUANTITYORDERED"],
    y=data["SALES"],
    hue=df["Hierarchical_Cluster"],
    palette="rainbow"
)

plt.title("Hierarchical Clustering")
plt.xlabel("Quantity Ordered")
plt.ylabel("Sales")
plt.show()

# ---------------------------------------
# 9. FINAL COMPARISON
# ---------------------------------------

print("\n===== FINAL RESULTS =====")

print("Number of Clusters:", k)

print("\nK-Means Cluster Counts:")
print(df["KMeans_Cluster"].value_counts().sort_index())

print("\nHierarchical Cluster Counts:")
print(df["Hierarchical_Cluster"].value_counts().sort_index())

print("\nClustering Completed Successfully!")
