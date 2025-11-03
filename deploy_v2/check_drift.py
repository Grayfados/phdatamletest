import pandas as pd
import os
from scipy.stats import ks_2samp, chi2_contingency

REFERENCE_DATA_PATH = os.path.join("..", "data", "kc_house_data.csv")
NEW_DATA_PATH = os.path.join("..", "data", "future_unseen_examples.csv")


NUMERIC_FEATURES_TO_CHECK = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
    'sqft_living15', 'sqft_lot15'
]

CATEGORICAL_FEATURES_TO_CHECK = [
    'zipcode', 'waterfront', 'view', 'condition', 'grade'
]

# P-value threshold. If p < 0.05, we assume drift has occurred.
P_VALUE_THRESHOLD = 0.05


def check_numerical_drift(ref_df, new_df, features):
    """Performs KS-Test on numerical features."""
    print("\n--- Checking Numerical Drift (KS-Test) ---")
    drift_found = False
    for col in features:
        if col not in ref_df.columns or col not in new_df.columns:
            print(f"Warning: Column {col} not found. Skipping.")
            continue

        # Perform the two-sample KS test
        ks_stat, p_value = ks_2samp(ref_df[col], new_df[col])

        if p_value < P_VALUE_THRESHOLD:
            print(f"[DRIFT DETECTED] Feature: '{col}'")
            print(f"  P-Value: {p_value:.4e} (Stat: {ks_stat:.4f})")
            drift_found = True
        else:
            print(f"[OK] Feature: '{col}' (p={p_value:.4f})")

    return drift_found


def check_categorical_drift(ref_df, new_df, features):
    """Performs Chi-Square Test on categorical features."""
    print("\n--- Checking Categorical Drift (Chi-Square) ---")
    drift_found = False
    for col in features:
        if col not in ref_df.columns or col not in new_df.columns:
            print(f"Warning: Column {col} not found. Skipping.")
            continue

        # Get value counts for both datasets
        ref_counts = ref_df[col].value_counts()
        new_counts = new_df[col].value_counts()

        # Combine into a contingency table
        contingency_table = pd.DataFrame({'reference': ref_counts, 'new': new_counts}).fillna(0)

        try:
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

            if p_value < P_VALUE_THRESHOLD:
                print(f"[DRIFT DETECTED] Feature: '{col}'")
                print(f"  P-Value: {p_value:.4e} (Stat: {chi2_stat:.4f})")
                drift_found = True
            else:
                print(f"[OK] Feature: '{col}' (p={p_value:.4f})")
        except ValueError:
            print(f"[ERROR] Could not run Chi-Square on '{col}'. Often due to too few samples.")

    return drift_found


def main():
    print(f"Loading reference data from: {REFERENCE_DATA_PATH}")
    df_ref = pd.read_csv(REFERENCE_DATA_PATH)

    print(f"Loading new data from: {NEW_DATA_PATH}")
    df_new = pd.read_csv(NEW_DATA_PATH)

    print(f"\nReference data shape: {df_ref.shape}")
    print(f"New data shape: {df_new.shape}")

    num_drift = check_numerical_drift(df_ref, df_new, NUMERIC_FEATURES_TO_CHECK)
    cat_drift = check_categorical_drift(df_ref, df_new, CATEGORICAL_FEATURES_TO_CHECK)

    print("\n--- Summary ---")
    if num_drift or cat_drift:
        print("ALERT: Data drift detected! A retraining pipeline should be triggered.")
    else:
        print("OK: No significant data drift detected.")


if __name__ == "__main__":
    main()