# File: deploy_v2/check_retraining.py
# (Code in English, as requested)
# This script simulates a retraining trigger by checking model performance
# on a new batch of "ground truth" data.

import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

MODEL_PATH = os.path.join("model", "model_v3.pkl")

DATA_PATH = os.path.join("..", "data", "kc_house_data.csv")
DEMOGRAPHICS_PATH = os.path.join("..", "data", "zipcode_demographics.csv")

R2_THRESHOLD = 0.75


def load_simulation_data(data_path, demo_path, fraction=0.1):
    """Loads the last 'fraction' of data to simulate a new batch."""
    print(f"Loading full dataset from {data_path}...")
    df_sales = pd.read_csv(data_path, dtype={'zipcode': str})

    df_sales['date'] = pd.to_datetime(df_sales['date'].str.split('T').str[0], format='%Y%m%d')
    df_sales = df_sales.sort_values(by='date')

    new_data_count = int(len(df_sales) * fraction)
    df_batch = df_sales.tail(new_data_count)

    print(f"Simulating new data with last {new_data_count} records.")

    df_demo = pd.read_csv(demo_path, dtype={'zipcode': str})
    df_merged = pd.merge(df_batch, df_demo, on='zipcode', how='left')

    y_true = df_merged.pop('price')
    X = df_merged.drop(['id', 'date'], axis=1, errors='ignore')

    X = X.fillna(0)

    return X, y_true


def main():
    print("--- Retraining Trigger Check ---")

    print(f"Loading deployed model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL: Model file not found at '{MODEL_PATH}'")
        print("Please run train_v3.py first.")
        return
    model = joblib.load(MODEL_PATH)

    X_new, y_new = load_simulation_data(DATA_PATH, DEMOGRAPHICS_PATH)

    print("Generating predictions on new data batch...")
    try:
        y_pred = model.predict(X_new)
    except Exception as e:
        print(f"FATAL: Error during prediction. {e}")
        print("This could be a 'train-serve skew' (data mismatch).")
        return

    r2 = r2_score(y_new, y_pred)
    rmse = np.sqrt(mean_squared_error(y_new, y_pred))

    print("\n--- Performance on New Data ---")
    print(f"  Current R²:   {r2:.4f}")
    print(f"  Business R² Threshold: {R2_THRESHOLD:.4f}")
    print(f"  Current RMSE: ${rmse:,.2f}")

    print("\n--- Decision ---")
    if r2 < R2_THRESHOLD:
        print(f"ALERT: Model performance ({r2:.4f}) is below threshold ({R2_THRESHOLD:.4f}).")
        print("ACTION: Triggering retraining pipeline!")
    else:
        print(f"OK: Model performance ({r2:.4f}) is above threshold.")
        print("ACTION: No retraining needed.")


if __name__ == "__main__":
    main()