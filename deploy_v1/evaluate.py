import pandas as pd
import joblib
import json
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
import numpy as np

# --- 1. Constants and Paths ---
DATA_FILE = 'data/kc_house_data.csv'
DEMOGRAPHICS_FILE = 'data/zipcode_demographics.csv'
MODEL_FILE = 'model/model.pkl'
FEATURES_FILE = 'model/model_features.json'

# --- 2. Load Model and Features ---
try:
    print(f"Loading model from {MODEL_FILE}...")
    model = joblib.load(MODEL_FILE)

    print(f"Loading features from {FEATURES_FILE}...")
    model_features = json.load(open(FEATURES_FILE))
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    exit(1)

# --- 3. Load and Prepare Data ---
print("Loading and preparing training data...")
try:
    df_house = pd.read_csv(DATA_FILE)
    df_demo = pd.read_csv(DEMOGRAPHICS_FILE)

    # Join demographics data to the house data
    df_full = pd.merge(df_house, df_demo, on='zipcode', how='left')

    # Define X (features) and y (target)
    y = df_full['price']

    # Filter the dataframe to only contain the features the model expects,
    # in the correct order
    X = df_full[model_features]

    # The original create_model.py script likely dropped NaNs. We replicate that here.
    # (A preprocessing pipeline would be a better solution)
    # We must drop NaNs from both X and y alignment.
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined[model_features]
    y = combined['price']

    print(f"Data ready. {len(X)} valid samples found.")

except FileNotFoundError as e:
    print(f"Error: Data file not found - {e}")
    exit(1)
except KeyError as e:
    print(f"Error: Expected column not found in data - {e}")
    exit(1)

# --- 4. Run Cross-Validation ---
print("\nStarting Cross-Validation (k=5)...")
print("This may take a few seconds...")

# We use 'neg_root_mean_squared_error' because scikit-learn aims to maximize scores.
scoring = {
    'r2': 'r2',
    'rmse': 'neg_root_mean_squared_error'
}

# We use k=5 (5 folds) as a reasonable default
try:
    cv_scores_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_scores_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')

    # Convert RMSE scores back to positive numbers
    cv_scores_rmse = -cv_scores_rmse

    # --- 5. Display Results ---
    print("\n--- Evaluation Results (Cross-Validation) ---")
    print(f"  Metrics for model: {MODEL_FILE}\n")

    print(f"  R² (R-squared):")
    print(f"    Mean:    {np.mean(cv_scores_r2):.4f}")
    print(f"    Std Dev: {np.std(cv_scores_r2):.4f}")
    print(f"    Values:  {[round(score, 4) for score in cv_scores_r2]}\n")

    print(f"  RMSE (Root Mean Squared Error):")
    print(f"    Mean:    ${np.mean(cv_scores_rmse):,.2f}")
    print(f"    Std Dev: ${np.std(cv_scores_rmse):,.2f}")
    print(f"    Values:  {[round(score, 2) for score in cv_scores_rmse]}\n")

    print("--- Generalization Analysis [Req. 3] ---")
    print("Answering: 'Does the model generalize well?' and 'Is it appropriately fit?'\n")

    if np.mean(cv_scores_r2) < 0.6:
        print(f"The mean R² of {np.mean(cv_scores_r2):.4f} is low. This indicates the model")
        print("explains less than 60% of the price variability.")
        print(">> Conclusion: The model is **underfitting**.")
        print("   It is too simple to capture the complexity of the data.")
    else:
        print(f"The mean R² of {np.mean(cv_scores_r2):.4f} is reasonable.")
        print(">> Conclusion: The model has basic predictive power, but")
        print("   it can likely be improved significantly.")

    print(f"\nThe mean RMSE of ${np.mean(cv_scores_rmse):,.2f} means the model's predictions")
    print("are, on average, off by this dollar amount.")
    print(f"The mean house price in the dataset is ${y.mean():,.2f}.")
    print(f"The error (RMSE) represents ~{(np.mean(cv_scores_rmse) / y.mean() * 100):.2f}% of the mean price.")


except Exception as e:
    print(f"An error occurred during cross-validation: {e}")