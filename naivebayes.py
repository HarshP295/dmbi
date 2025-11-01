




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.impute import SimpleImputer # New import for handling NaNs


# --- CONFIGURATION: Change these for a new dataset ---
# ⚠️ 1. Change the file name
# Note: Based on your traceback, you were trying to use a file named 'breast_cancer.csv'
DATASET_FILE = 'Iris.csv'


# ⚠️ 2. Change the name of the target/label column
# Note: Based on your traceback, the target column is 'diagnosis'
TARGET_COLUMN = 'Species'


# ⚠️ 3. Change the name of the unique ID column to drop (or set to None if no ID column exists)
# Note: Based on your traceback, the ID column is 'id'
ID_COLUMN_TO_DROP = 'id'
# --- END CONFIGURATION ---




# --- 1. Preprocess data. Split data into train and test set ---


# Load the dataset
print("Loading data...")
try:
    df = pd.read_csv(DATASET_FILE)
    print(f"Successfully loaded '{DATASET_FILE}'.")
except FileNotFoundError:
    print(f"Error: '{DATASET_FILE}' not found. Make sure the file is in the same directory.")
    exit()


# Drop the specified ID column if it exists
if ID_COLUMN_TO_DROP and ID_COLUMN_TO_DROP in df.columns:
    df = df.drop(ID_COLUMN_TO_DROP, axis=1)
    print(f"Dropped ID column: '{ID_COLUMN_TO_DROP}'")
else:
    print("No specified ID column to drop, or column not found.")
   
# Clean up any residual unnamed columns (often happens with CSV exports)
# Based on your traceback, there is an 'Unnamed: 32' column which is causing issues.
unnamed_cols = [col for col in df.columns if 'Unnamed:' in col]
if unnamed_cols:
    df = df.drop(columns=unnamed_cols)
    print(f"Dropped spurious columns: {unnamed_cols}")




# Separate features (X) and target (y)
# Features (X) will now be all columns *except* the target column
try:
    y = df[TARGET_COLUMN]
    X = df.drop(TARGET_COLUMN, axis=1)
    print(f"Features set (X) columns count: {len(X.columns)}")
    print(f"Target variable (y): '{TARGET_COLUMN}'")
except KeyError:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataset.")
    exit()


# --- NEW STEP: Handle Missing Values (NaN) via Mean Imputation ---
if X.isnull().sum().any():
    print(f"\nHandling {X.isnull().sum().sum()} missing values using SimpleImputer (mean strategy)...")
   
    # Initialize the imputer to fill missing values with the mean of the column
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
   
    # Fit and transform the features data
    X_imputed = imputer.fit_transform(X)
   
    # Convert back to DataFrame, keeping column names
    X = pd.DataFrame(X_imputed, columns=X.columns)
    print("Missing values imputation complete.")
else:
    print("\nNo missing values found in features (X). Skipping imputation.")
# ------------------------------------------------------------------




# Split the data into training (80%) and testing (20%) sets
print("Splitting data into 80% training and 20% testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Display shapes of the split data
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print("-" * 50)


# --- 2. Build a Classification model using the inbuilt library function on training data ---


# Initialize the Gaussian Naive Bayes model
nb_model = GaussianNB()


# Train the model on the training data
print("Training Gaussian Naive Bayes model...")
nb_model.fit(X_train, y_train)
print("Training complete.")
print("-" * 50)


# Make predictions on the test set
y_pred = nb_model.predict(X_test)


# --- 3. Calculate metrics based on test data using an inbuilt function ---


# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Accuracy: {accuracy:.4f}")


# Generate a Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Generate Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# Interpretation of Results
print("\n--- Model Evaluation Summary ---")
print(f"The Naive Bayes model achieved an accuracy of {accuracy:.4f} on the test set.")
print("It performs very well, demonstrating its suitability for this type of feature data after imputation.")



