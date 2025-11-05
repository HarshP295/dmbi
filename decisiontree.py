# --------------------------------------------------------------
# 🌸 Decision Tree Classifier (Iris Dataset, Colab-Ready Version)
# --------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from google.colab import files  # ✅ For interactive upload

# --- 1. Upload the Iris dataset file ---
print("📂 Please upload your Iris dataset (e.g., Iris.csv)...")
uploaded = files.upload()

# Automatically grab the uploaded filename
DATASET_FILE = list(uploaded.keys())[0]
print(f"✅ File '{DATASET_FILE}' uploaded successfully.\n")


# --- 2. Configuration ---
TARGET_COLUMN = 'Species'   # Label column
ID_COLUMN_TO_DROP = 'Id'    # Column to drop (if exists)


# --- 3. Load and preprocess data ---
print("🔍 Loading dataset...")
df = pd.read_csv(DATASET_FILE)
print(f"✅ Loaded '{DATASET_FILE}' successfully with shape {df.shape}\n")

# Drop ID column if it exists
if ID_COLUMN_TO_DROP in df.columns:
    df.drop(ID_COLUMN_TO_DROP, axis=1, inplace=True)
    print(f"🗑️ Dropped ID column: '{ID_COLUMN_TO_DROP}'")
else:
    print("ℹ️ No ID column found or nothing to drop.\n")

# Split into features (X) and target (y)
try:
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    print(f"📊 Features: {list(X.columns)}")
    print(f"🎯 Target column: '{TARGET_COLUMN}'\n")
except KeyError:
    print(f"❌ Error: Target column '{TARGET_COLUMN}' not found in dataset.")
    exit()


# --- 4. Train-Test Split ---
print("✂️ Splitting data into 80% train and 20% test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ X_train: {X_train.shape}, X_test: {X_test.shape}\n")


# --- 5. Train Decision Tree ---
print("🌳 Training Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("✅ Training complete.\n")


# --- 6. Evaluate Model ---
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"📈 Accuracy: {accuracy:.4f}\n")
print("🧾 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🔢 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n✅ Model evaluation done.\n")


# --- 7. Visualize Decision Tree ---
plt.figure(figsize=(20, 10))
feature_names = X.columns.tolist()
class_names = dt_model.classes_.astype(str).tolist()

plot_tree(dt_model, feature_names=feature_names, class_names=class_names, filled=True)
plt.title(f"Decision Tree Visualization - {DATASET_FILE}")
PLOT_FILENAME = "decision_tree_output.png"
plt.savefig(PLOT_FILENAME)
plt.show()

print(f"🖼️ Decision Tree plot saved as '{PLOT_FILENAME}'\n")
print("✅ All steps completed successfully.")
