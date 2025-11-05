# --- IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from google.colab import files  # ✅ for file upload in Colab

# --------------------------------------------------------------------------------
# --- FILE UPLOAD (Colab) ---
# --------------------------------------------------------------------------------
print("📂 Please upload your CSV dataset (e.g., Iris.csv)...")
uploaded = files.upload()

# Get the uploaded file name
DATASET_FILE = list(uploaded.keys())[0]
print(f"✅ Successfully uploaded: {DATASET_FILE}")

# --------------------------------------------------------------------------------
# --- CONFIGURATION ---
# --------------------------------------------------------------------------------
# FEATURES_TO_DROP = ['Id', 'Species']   # Drop irrelevant columns
# N_CLUSTERS = 3                         # Default number of clusters
# LINKAGE = 'ward'                       # Linkage type for Agglomerative Clustering
# VISUALIZATION_FEATURE_1 = 'PetalLengthCm'
# VISUALIZATION_FEATURE_2 = 'PetalWidthCm'

# # SCENARIO 2: Mall Customer Dataset (Typically 5 optimal clusters)

FEATURES_TO_DROP = ['CustomerID', 'Gender', 'Age'] # Focus on Income and Score
N_CLUSTERS = 5
LINKAGE = 'ward'
VISUALIZATION_FEATURE_1 = 'Annual Income (k$)'
VISUALIZATION_FEATURE_2 = 'Spending Score (1-100)'


# --------------------------------------------------------------------------------
# --- 1. LOAD & PREPROCESS DATA ---
# --------------------------------------------------------------------------------
print("\nLoading data...")
try:
    df = pd.read_csv(DATASET_FILE)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    raise SystemExit

# Drop specified columns
X = df.drop(columns=[col for col in FEATURES_TO_DROP if col in df.columns], errors='ignore')

# Encode categorical features
print("\n🔤 Encoding categorical features (if any)...")
X = pd.get_dummies(X, drop_first=True)

# Handle missing values
if X.isnull().sum().any():
    print(f"⚠️ Found {X.isnull().sum().sum()} missing values. Filling with column means...")
    imputer = SimpleImputer(strategy='mean')
    X[:] = imputer.fit_transform(X)
else:
    print("✅ No missing values found.")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Features scaled successfully.")
print("-" * 50)

# --------------------------------------------------------------------------------
# --- 2. BUILD CLUSTERING MODEL ---
# --------------------------------------------------------------------------------
print(f"🚀 Running Agglomerative Clustering with {N_CLUSTERS} clusters and '{LINKAGE}' linkage...")
agg_clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage=LINKAGE)
labels = agg_clustering.fit_predict(X_scaled)
df['Cluster'] = labels
print("✅ Clustering complete.")
print("-" * 50)

# --------------------------------------------------------------------------------
# --- 3. PERFORMANCE EVALUATION ---
# --------------------------------------------------------------------------------
n_clusters_found = len(np.unique(labels))
if n_clusters_found > 1:
    score = silhouette_score(X_scaled, labels)
    print(f"📊 Silhouette Score: {score:.4f}")
else:
    print("⚠️ Cannot calculate Silhouette Score (only one cluster found).")

print("\n--- Summary ---")
print(f"Total clusters formed: {n_clusters_found}")
if n_clusters_found > 1:
    print(f"Silhouette Score of {score:.4f} indicates how well-separated the clusters are.")

# --------------------------------------------------------------------------------
# --- 4. VISUALIZATION ---
# --------------------------------------------------------------------------------
try:
    plt.figure(figsize=(10, 7))
    plot_df = df[[VISUALIZATION_FEATURE_1, VISUALIZATION_FEATURE_2, 'Cluster']].copy()
    plot_df['Cluster'] = plot_df['Cluster'].astype('category')

    sns.scatterplot(
        x=VISUALIZATION_FEATURE_1,
        y=VISUALIZATION_FEATURE_2,
        hue='Cluster',
        data=plot_df,
        palette='Spectral',
        s=100,
        style='Cluster'
    )

    plt.title(f"Agglomerative Clustering ({N_CLUSTERS} Clusters, Linkage='{LINKAGE}')")
    plt.xlabel(VISUALIZATION_FEATURE_1)
    plt.ylabel(VISUALIZATION_FEATURE_2)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"❌ Error during visualization: {e}")
