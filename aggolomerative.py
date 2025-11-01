import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


# --------------------------------------------------------------------------------
# --- CONFIGURATION: Change these variables for a new dataset ---
# --------------------------------------------------------------------------------


# SCENARIO 1: Iris Dataset (3 natural clusters)
DATASET_FILE = 'Iris.csv'
FEATURES_TO_DROP = ['Id', 'Species'] # Drop ID and the known label
N_CLUSTERS = 3
LINKAGE = 'ward' # Minimizes variance within each cluster
VISUALIZATION_FEATURE_1 = 'PetalLengthCm'
VISUALIZATION_FEATURE_2 = 'PetalWidthCm'




# # SCENARIO 2: Mall Customer Dataset (Typically 5 optimal clusters)
# DATASET_FILE = 'Mall_Customers.csv'
# FEATURES_TO_DROP = ['CustomerID', 'Gender', 'Age'] # Focus on Income and Score
# N_CLUSTERS = 5
# LINKAGE = 'ward'
# VISUALIZATION_FEATURE_1 = 'Annual Income (k$)'
# VISUALIZATION_FEATURE_2 = 'Spending Score (1-100)'




# --------------------------------------------------------------------------------
# --- END CONFIGURATION ---
# --------------------------------------------------------------------------------


# --- 1. Preprocess data (Scaling is essential for Agglomerative Clustering) ---


# Load the dataset
print("Loading data...")
try:
    df = pd.read_csv(DATASET_FILE)
    print(f"Successfully loaded '{DATASET_FILE}'.")
except FileNotFoundError:
    print(f"Error: '{DATASET_FILE}' not found. Make sure the file is in the same directory.")
    exit()


# Drop specified features
X = df.drop(columns=[col for col in FEATURES_TO_DROP if col in df.columns], errors='ignore')


# Handle categorical features (One-Hot Encoding)
print("\nEncoding categorical features...")
X = pd.get_dummies(X, drop_first=True)


# Impute missing values (just in case)
if X.isnull().sum().any():
    print(f"Handling {X.isnull().sum().sum()} missing values using SimpleImputer (mean strategy)...")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X.values)
    X = pd.DataFrame(X_imputed, columns=X.columns)
    print("Missing values imputation complete.")
else:
    print("No missing values found. Skipping imputation.")




# Feature Scaling (mandatory for distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data scaled using StandardScaler.")
print("-" * 50)




# --- 2. Build a Clustering model using the inbuilt library function ---


# Initialize Agglomerative Clustering using configurable parameters
print(f"Building Agglomerative Clustering model with n_clusters={N_CLUSTERS} and linkage='{LINKAGE}'...")
agg_clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage=LINKAGE)


# Fit the model (find the clusters) and generate labels
labels = agg_clustering.fit_predict(X_scaled)
df['Cluster'] = labels # Add labels back to the original DataFrame for plotting
print("Clustering complete.")
print("-" * 50)




# --- 3. Determine Performance parameters ---


# Number of clusters found
n_clusters_found = len(np.unique(labels))


# Calculate Silhouette Score (A measure of cluster quality)
if n_clusters_found > 1:
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score (A measure of cluster separation): {score:.4f}")
else:
    print("Silhouette Score cannot be calculated (less than 2 clusters found).")


# Interpretation of Results
print("\n--- Model Evaluation Summary ---")
print(f"Agglomerative Clustering found {n_clusters_found} clusters.")
if n_clusters_found > 1:
    print(f"A Silhouette Score of {score:.4f} confirms the quality of the partition with the chosen parameters.")




# --- 4. Visualize the Clusters ---
try:
    plt.figure(figsize=(10, 7))
   
    # Merge the original features needed for visualization with the new cluster labels
    plot_df = df[[VISUALIZATION_FEATURE_1, VISUALIZATION_FEATURE_2, 'Cluster']].copy()


    # Convert Cluster to categorical for plotting
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
    plt.title(f'Agglomerative Clustering: {N_CLUSTERS} Clusters (Linkage: {LINKAGE})')
    plt.xlabel(VISUALIZATION_FEATURE_1)
    plt.ylabel(VISUALIZATION_FEATURE_2)
    plt.grid(True)
   
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("agglomerative_cluster_visualization.png")
    print("\nAgglomerative cluster visualization saved as 'agglomerative_cluster_visualization.png'")
   
except Exception as e:
    print(f"\nError during visualization: {e}")



