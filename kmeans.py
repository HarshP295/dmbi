# ---------------------------------------------------------------
# K-Means Clustering with Colab File Upload Support
# ---------------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from google.colab import files  # ✅ For file upload in Colab

# ---------------------------------------------------------------
# Step 1: File Upload (Colab Dialog)
# ---------------------------------------------------------------

print("📂 Please upload your dataset CSV file...")
uploaded = files.upload()

if len(uploaded) == 0:
    print("❌ No file uploaded. Please try again.")
    exit()

DATASET_FILE = list(uploaded.keys())[0]
print(f"✅ File uploaded successfully: {DATASET_FILE}\n")

# ---------------------------------------------------------------
# Step 2: Configuration — you can change these for your dataset
# ---------------------------------------------------------------

# Example 1: Iris Dataset
FEATURES_TO_DROP = ['Id', 'Species']
N_CLUSTERS = 3
VISUALIZATION_FEATURE_1 = 'SepalLengthCm'
VISUALIZATION_FEATURE_2 = 'PetalLengthCm'

# # Example 2: Mall Customers Dataset
# FEATURES_TO_DROP = ['CustomerID', 'Gender', 'Age']
# N_CLUSTERS = 5
# VISUALIZATION_FEATURE_1 = 'Annual Income (k$)'
# VISUALIZATION_FEATURE_2 = 'Spending Score (1-100)'

# ---------------------------------------------------------------
# Step 3: Load and Preprocess Data
# ---------------------------------------------------------------

print("🔄 Loading dataset...")
df = pd.read_csv(DATASET_FILE)
print(f"✅ Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.\n")

# Drop specified columns
X = df.drop(columns=[col for col in FEATURES_TO_DROP if col in df.columns], errors='ignore')

# Convert categorical to numeric (dummy encoding)
X = pd.get_dummies(X, drop_first=True)

# Handle missing values
if X.isnull().sum().any():
    print(f"⚠️ Found {X.isnull().sum().sum()} missing values — filling with column means...")
    imputer = SimpleImputer(strategy='mean')
    X[:] = imputer.fit_transform(X)
else:
    print("✅ No missing values found.")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Data scaled successfully.\n")

# ---------------------------------------------------------------
# Step 4: Apply K-Means Clustering
# ---------------------------------------------------------------

print(f"🚀 Running K-Means with {N_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', random_state=42, n_init='auto')
kmeans.fit(X_scaled)
labels = kmeans.labels_
df['Cluster'] = labels
print("✅ Clustering complete.\n")

# ---------------------------------------------------------------
# Step 5: Evaluate Model
# ---------------------------------------------------------------

inertia = kmeans.inertia_
print(f"📊 Inertia (WCSS): {inertia:.4f}")

if N_CLUSTERS > 1:
    silhouette = silhouette_score(X_scaled, labels)
    print(f"📈 Silhouette Score: {silhouette:.4f}\n")

print("--- Interpretation ---")
print("• Lower inertia → tighter clusters")
print("• Silhouette Score near 1 → well-separated clusters\n")

# ---------------------------------------------------------------
# Step 6: Visualization
# ---------------------------------------------------------------

try:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=VISUALIZATION_FEATURE_1,
        y=VISUALIZATION_FEATURE_2,
        hue='Cluster',
        data=df,
        palette='viridis',
        s=100,
        style='Cluster'
    )
    plt.title(f'K-Means Clustering ({N_CLUSTERS} Clusters)')
    plt.xlabel(VISUALIZATION_FEATURE_1)
    plt.ylabel(VISUALIZATION_FEATURE_2)
    plt.grid(True)

    # Attempt to overlay centroids (approximation)
    centers_scaled = kmeans.cluster_centers_
    try:
        idx1 = X.columns.get_loc(VISUALIZATION_FEATURE_1)
        idx2 = X.columns.get_loc(VISUALIZATION_FEATURE_2)
        plt.scatter(
            X_scaled[:, idx1],
            X_scaled[:, idx2],
            marker='X',
            s=250,
            color='red',
            label='Centroids (approx)',
            edgecolors='black'
        )
    except KeyError:
        print("\n⚠️ Visualization features not found in final data (possibly dropped or encoded). Skipping centroids plot.")
        pass

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"❌ Error during visualization: {e}")
