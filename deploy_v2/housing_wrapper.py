# File: deploy_v2/housing_wrapper.py (WITH DEBUG LOGS)
# (Code in English, as requested)

import mlflow
import joblib
import redis
import json
import pandas as pd
import numpy as np


# --- Feature Engineering Function ---
# (Unchanged, but must be present)
def engineer_features(df):
    """Creates new time-based features from the date column."""
    df_fe = df.copy()

    try:
        df_fe['date'] = pd.to_datetime(df_fe['date'].str.split('T').str[0], format='%Y%m%d')
    except Exception as e:
        print(f"DEBUG (engineer_features): Warning: Could not parse all dates. Error: {e}")
        df_fe['date'] = pd.to_datetime(df_fe['date'], errors='coerce')

    df_fe['sale_year'] = df_fe['date'].dt.year
    df_fe['sale_month'] = df_fe['date'].dt.month
    df_fe['sale_dayofweek'] = df_fe['date'].dt.dayofweek
    df_fe['house_age'] = df_fe['sale_year'] - df_fe['yr_built']
    df_fe['yrs_since_renovated'] = np.where(
        df_fe['yr_renovated'] == 0,
        df_fe['house_age'],
        df_fe['sale_year'] - df_fe['yr_renovated']
    )
    df_fe = df_fe.drop(['date', 'yr_built', 'yr_renovated', 'id'], axis=1, errors='ignore')
    return df_fe


# --- MLflow Custom Model Wrapper ---
class HousingModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """Called when loading the model for serving."""
        print("--- [DEBUG] LOAD_CONTEXT START ---")
        print("DEBUG: Loading model_pipeline artifact...")
        self.model = joblib.load(context.artifacts["model_pipeline"])

        # Get the list of features the model *actually* expects
        try:
            # Get features from the preprocessor step of the pipeline
            self.expected_features = self.model.named_steps['preprocessor'].feature_names_in_
            print(f"DEBUG: Successfully loaded {len(self.expected_features)} expected feature names from pipeline.")
        except Exception as e:
            print(f"DEBUG: WARNING! Could not get feature_names_in_ from pipeline. {e}")
            self.expected_features = []

        print("DEBUG: Connecting to Redis (localhost:6379)...")
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            print("DEBUG: Redis connection successful.")
        except redis.exceptions.ConnectionError as e:
            print(f"FATAL (LOAD_CONTEXT): Error connecting to Redis: {e}")
            self.redis_client = None
        print("--- [DEBUG] LOAD_CONTEXT END ---")

    def predict(self, context, model_input):
        """Called for inference."""

        # --- [ DEBUG 1: INPUT ] ---
        print("\n--- [DEBUG] PREDICT START ---")
        print(f"DEBUG (1. Input): Received model_input with shape {model_input.shape}")
        print(f"DEBUG (1. Input): Columns: {list(model_input.columns)}")

        if self.redis_client is None:
            raise Exception("Redis connection not available. Service is unhealthy.")

        df = model_input.copy()

        # --- [ DEBUG 2: REDIS LOOKUP ] ---
        def get_redis_data(zipcode):
            clean_zip = None
            data_json = None
            try:
                clean_zip = str(int(float(zipcode)))
                data_json = self.redis_client.get(clean_zip)

                if data_json:
                    if not hasattr(self, '_printed_redis_success'):
                        print(f"DEBUG (2. Redis): SUCCESS. Found data for zip {clean_zip}. Length: {len(data_json)}")
                        self._printed_redis_success = True  # Print only once
                    return json.loads(data_json)
                else:
                    if not hasattr(self, '_printed_redis_fail'):
                        print(
                            f"DEBUG (2. Redis): FAILED. No data found for zip {clean_zip} (Input: {zipcode}). Is Redis populated?")
                        self._printed_redis_fail = True  # Print only once
            except Exception as e:
                print(f"DEBUG (2. Redis): ERROR processing zip {zipcode}. Error: {e}")
            return {}  # Return empty dict on failure

        print("DEBUG (2. Redis): Fetching demographics from Redis...")
        demographics = df['zipcode'].apply(get_redis_data)
        df_demo = pd.DataFrame.from_records(demographics)

        # --- [ DEBUG 3: DEMO DATAFRAME ] ---
        print(f"DEBUG (3. Demo DF): Created df_demo with shape {df_demo.shape}")
        if df_demo.empty:
            print("DEBUG (3. Demo DF): ERROR! df_demo is EMPTY. Redis lookup failed for all inputs.")
        else:
            print(f"DEBUG (3. Demo DF): Columns: {list(df_demo.columns)}")

        # --- [ DEBUG 4: COMBINED DATAFRAME ] ---
        df_demo_clean = df_demo.drop('zipcode', axis=1, errors='ignore')
        df_combined = pd.concat([df.reset_index(drop=True), df_demo_clean.reset_index(drop=True)], axis=1)
        print(f"DEBUG (4. Combined DF): Created df_combined with shape {df_combined.shape}")
        print(f"DEBUG (4. Combined DF): Columns: {list(df_combined.columns)}")

        # --- [ DEBUG 5: FEATURE ENGINEERING ] ---
        print("DEBUG (5. FE): Performing feature engineering...")
        df_final_features = engineer_features(df_combined)
        print(f"DEBUG (5. FE): Created df_final_features with shape {df_final_features.shape}")
        print(f"DEBUG (5. FE): Columns: {list(df_final_features.columns)}")

        # --- [ DEBUG 6: PREDICTION & ERROR CATCHING ] ---
        try:
            print("DEBUG (6. Predict): Calling self.model.predict()...")
            prediction = self.model.predict(df_final_features)

            print("DEBUG (6. Predict): Prediction successful.")
            print("--- [DEBUG] PREDICT END (SUCCESS) ---")
            return prediction

        except KeyError as e:
            print("\n--- [DEBUG] KEYERROR CAUGHT! ---")
            print(f"ERROR: {e}")
            print("\nThe model pipeline failed because the DataFrame is missing columns.")

            expected_cols = set(self.expected_features)
            actual_cols = set(df_final_features.columns)

            missing_in_df = expected_cols - actual_cols
            extra_in_df = actual_cols - expected_cols

            print(f"\nDEBUG (6. Predict): COLUMN MISMATCH:")
            print(f"  Missing (Model expected, but DF didn't have): {missing_in_df}")
            print(f"  Extra (DF had, but Model didn't expect): {extra_in_df}")

            print("--- [DEBUG] PREDICT END (WITH KEYERROR) ---")
            raise e  # Re-raise the error
        except Exception as e:
            print(f"\n--- [DEBUG] UNEXPECTED ERROR CAUGHT! ---")
            print(f"ERROR: {e}")
            print("--- [DEBUG] PREDICT END (WITH UNEXPECTED ERROR) ---")
            raise e


# --- CRITICAL LINE ---
# This tells MLflow what object *is* the model
print("Registering HousingModelWrapper with mlflow.models.set_model()")
mlflow.models.set_model(HousingModelWrapper())