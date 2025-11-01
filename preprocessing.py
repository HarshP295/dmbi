import pandas as pd
import numpy as np

# Load your dataset
# df = pd.read_csv('path_to_your_dataset.csv')

# Display data info
print("Original data shape:", df.shape)
print(df.info())
print(df.head())

# 1. Drop duplicate rows
df = df.drop_duplicates()

# 2. Handle missing values
# Drop rows with all fields missing
df = df.dropna(how='all')

# For columns with too many missing values, drop column (example threshold 70% missing)
missing_pct = df.isnull().mean()
cols_to_drop = missing_pct[missing_pct > 0.7].index
df = df.drop(columns=cols_to_drop)

# For columns with moderate missing, fill with median (numeric) or mode (categorical)
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# 3. General type conversion (example: convert 'date' columns to datetime)
for col in df.columns:
    if 'date' in col.lower():
        df[col] = pd.to_datetime(df[col], errors='ignore')

# 4. Lowercase string columns and strip whitespace
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.lower().str.strip()

# 5. (Optional) Remove outliers from numerical columns (using Z-score)
from scipy.stats import zscore
for col in df.select_dtypes(include=[np.number]).columns:
    df = df[(np.abs(zscore(df[col])) < 3) | (df[col].isnull())]

print("Data after cleaning:", df.shape)
print(df.head())



