# --------------------------------------------------------------------------------
# --- ASSOCIATION RULE MINING (Apriori) with KaggleHub Auto-Download ---
# --------------------------------------------------------------------------------

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --------------------------------------------------------------------------------
# --- DOWNLOAD DATASET FROM KAGGLE ---
# --------------------------------------------------------------------------------

print("📦 Downloading 'Groceries Dataset' from Kaggle using KaggleHub...")
path = kagglehub.dataset_download("heeraldedhia/groceries-dataset")
print("✅ Dataset downloaded to:", path)

# Find the correct file (Groceries_dataset.csv)
DATASET_FILE = os.path.join(path, "Groceries_dataset.csv")

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"❌ Could not find 'Groceries_dataset.csv' in {path}")

# --------------------------------------------------------------------------------
# --- CONFIGURATION ---
# --------------------------------------------------------------------------------

TRANSACTION_ID_COLUMNS = ['Member_number', 'Date']
ITEM_COLUMN = 'itemDescription'  # Kaggle file uses lowercase 'itemDescription'

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.1

# --------------------------------------------------------------------------------
# --- STEP 1: LOAD AND CLEAN DATA ---
# --------------------------------------------------------------------------------

print(f"\n📂 Loading dataset: {DATASET_FILE}...")
df = pd.read_csv(DATASET_FILE)

# Fix column name case sensitivity
if 'itemDescription' in df.columns and 'ItemDescription' not in df.columns:
    df = df.rename(columns={'itemDescription': 'ItemDescription'})
    print("🔧 Renamed 'itemDescription' → 'ItemDescription'")

ITEM_COLUMN = 'ItemDescription'

# Drop duplicates within the same transaction
df = df.drop_duplicates(subset=TRANSACTION_ID_COLUMNS + [ITEM_COLUMN], keep='first')
print("✅ Duplicate items per transaction removed.")

print(f"📊 Total Transactions: {len(df)}")
print(f"🛒 Total Unique Items: {df[ITEM_COLUMN].nunique()}")
print("-" * 70)

# --------------------------------------------------------------------------------
# --- STEP 2: CREATE ONE-HOT ENCODED MATRIX ---
# --------------------------------------------------------------------------------

print("🔄 Creating one-hot encoded transaction matrix...")

basket_sets = (
    df.groupby(TRANSACTION_ID_COLUMNS)[ITEM_COLUMN]
      .apply(lambda x: pd.Series(1, index=x))
      .unstack(fill_value=0)
)

basket_sets.columns.name = None
basket_sets.index.names = TRANSACTION_ID_COLUMNS
basket_sets = basket_sets.applymap(lambda x: 1 if x > 0 else 0)

print(f"✅ One-Hot Encoded Matrix Shape: {basket_sets.shape}")
print("-" * 70)

# --------------------------------------------------------------------------------
# --- STEP 3: RUN APRIORI ALGORITHM ---
# --------------------------------------------------------------------------------

print(f"⚙️ Running Apriori (min_support = {MIN_SUPPORT})...")
frequent_itemsets = apriori(basket_sets, min_support=MIN_SUPPORT, use_colnames=True)
print(f"✅ Found {len(frequent_itemsets)} frequent itemsets.")
print("-" * 70)

# --------------------------------------------------------------------------------
# --- STEP 4: GENERATE ASSOCIATION RULES ---
# --------------------------------------------------------------------------------

print(f"📈 Generating rules (min_confidence = {MIN_CONFIDENCE})...")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules = rules.sort_values(by=['lift', 'confidence'], ascending=False).reset_index(drop=True)

print(f"✅ Generated {len(rules)} association rules.")
print("\n🔝 Top 10 Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
print("-" * 70)

# --------------------------------------------------------------------------------
# --- STEP 5: VISUALIZATION ---
# --------------------------------------------------------------------------------

if not rules.empty:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(rules['support'], rules['confidence'], c=rules['lift'],
                          s=rules['lift']*50, alpha=0.7, cmap='viridis')

    plt.xlabel("Support (Frequency)")
    plt.ylabel("Confidence (Predictive Power)")
    plt.title("Association Rules: Confidence vs Support (Color = Lift)")

    cbar = plt.colorbar(scatter)
    cbar.set_label('Lift (Association Strength)')

    plt.axhline(MIN_CONFIDENCE, color='r', linestyle='--', linewidth=1, label=f'Min Confidence = {MIN_CONFIDENCE}')
    plt.axvline(MIN_SUPPORT, color='b', linestyle='--', linewidth=1, label=f'Min Support = {MIN_SUPPORT}')

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.savefig("apriori_rules_visualization.png")

    print("\n📊 Visualization saved as 'apriori_rules_visualization.png'")

# --------------------------------------------------------------------------------
# --- SUMMARY ---
# --------------------------------------------------------------------------------

print("\n--- 📘 SUMMARY ---")
print("✅ High Lift (>1) suggests strong item associations.")
if not rules.empty:
    print(f"Highest Lift observed: {rules['lift'].max():.3f}")
    print("Focus on rules with both high Lift and Confidence for actionable insights.")
