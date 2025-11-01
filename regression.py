



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.impute import SimpleImputer


# --------------------------------------------------------------------------------
# --- CONFIGURATION: Change these three variables for a new dataset ---
# --------------------------------------------------------------------------------


# # SCENARIO 1: Iris Dataset (Simple numerical data)
# DATASET_FILE = 'Iris.csv'
# TARGET_COLUMN = 'PetalLengthCm'
# # We keep 'Species' (categorical) for automatic encoding, drop 'Id'.
# COLUMNS_TO_DROP = ['Id']




# # SCENARIO 2: Insurance Dataset (Mixed numerical and categorical data)
DATASET_FILE = 'insurance.csv'
TARGET_COLUMN = 'charges'
# No ID column or other features to drop, as the model handles categorical features now.
COLUMNS_TO_DROP = []




# --------------------------------------------------------------------------------
# --- END CONFIGURATION ---
# --------------------------------------------------------------------------------




# --- 1. Preprocess data. Split data into train and test set ---


# Load the dataset
print("Loading data...")
try:
    df = pd.read_csv(DATASET_FILE)
    print(f"Successfully loaded '{DATASET_FILE}'.")
except FileNotFoundError:
    print(f"Error: '{DATASET_FILE}' not found. Make sure the file is in the same directory.")
    exit()


# Drop specified columns
for col in COLUMNS_TO_DROP:
    if col in df.columns:
        df = df.drop(col, axis=1)
        print(f"Dropped column: '{col}'")




# Separate features (X) and target (y)
try:
    y = df[TARGET_COLUMN]
    X = df.drop(TARGET_COLUMN, axis=1)
   
    print(f"Target variable (y): '{TARGET_COLUMN}'")


except KeyError:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataset.")
    exit()




# --- NEW STEP A: Handle Categorical Features (One-Hot Encoding) ---
# This step is critical for datasets like Insurance
print("\nPerforming One-Hot Encoding on categorical features...")
X = pd.get_dummies(X, drop_first=True)
print(f"Feature columns after encoding: {len(X.columns)}")
# ------------------------------------------------------------------




# --- NEW STEP B: Handle Missing Values (NaN) via Mean Imputation ---
# This ensures all input features are numerical and non-missing
if X.isnull().sum().any():
    print(f"Handling {X.isnull().sum().sum()} missing values using SimpleImputer (mean strategy)...")
   
    # Initialize the imputer to fill missing values with the mean of the column
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
   
    # Fit and transform the features data
    # We use .values here to pass the numpy array to the imputer
    X_imputed = imputer.fit_transform(X.values)
   
    # Convert back to DataFrame, keeping column names
    X = pd.DataFrame(X_imputed, columns=X.columns)
    print("Missing values imputation complete.")
else:
    print("No missing values found in features (X). Skipping imputation.")
# ------------------------------------------------------------------




# Split the data into training (80%) and testing (20%) sets
print("Splitting data into 80% training and 20% testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Display shapes of the split data
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print("-" * 50)


# --- 2. Build Regression model using inbuilt library function on training data ---


# Initialize the Linear Regression model
reg_model = LinearRegression()


# Train the model on the training data
print(f"Training Linear Regression model to predict {TARGET_COLUMN}...")
reg_model.fit(X_train, y_train)
print("Training complete.")
print("-" * 50)


# Make predictions on the test set
y_pred = reg_model.predict(X_test)


# --- 3. Calculate metrics based on test data using inbuilt function ---


# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # Root Mean Squared Error for easier interpretation
r2 = r2_score(y_test, y_pred)


print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")


# Interpretation of Results
print("\n--- Model Evaluation Summary ---")
print(f"The model explains approximately {r2*100:.2f}% of the variance in {TARGET_COLUMN} (R-squared).")
print(f"The average magnitude of error in prediction (MAE) is {mae:.4f}.")



