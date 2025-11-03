# File: deploy_v2/housing_wrapper_v3.py (With Dtype Fix)
# (Code in English, as requested)

import mlflow
import joblib
import redis
import json
import pandas as pd
import numpy as np


# --- Feature Engineering Function (REMOVED) ---

# --- MLflow Custom Model Wrapper ---
class HousingModelWrapperV3(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """Called when loading the model for serving."""
        print("--- [V3 Wrapper] LOAD_CONTEXT START ---")
        print("DEBUG: Loading model_v3_pipeline artifact...")
        self.model = joblib.load(context.artifacts["model_v3_pipeline"])

        print("DEBUG: Connecting to Redis (localhost:6379)...")
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            print("DEBUG: Redis connection successful.")
        except redis.exceptions.ConnectionError as e:
            print(f"FATAL (LOAD_CONTEXT): Error connecting to Redis: {e}")
            self.redis_client = None

    def predict(self, context, model_input):
        """Called for inference."""
        print("\n--- [V3 Wrapper] PREDICT START ---")
        print(f"DEBUG (1. Input): Received model_input with shape {model_input.shape}")

        if self.redis_client is None:
            raise Exception("Redis connection not available. Service is unhealthy.")

        df = model_input.copy()

        # 1. Get demographic data from Redis
        def get_redis_data(zipcode):
            try:
                clean_zip = str(int(float(zipcode)))
                data_json = self.redis_client.get(clean_zip)
                if data_json:
                    return json.loads(data_json)
            except Exception:
                pass
            return {}

        print("DEBUG (2. Redis): Fetching demographics from Redis...")
        demographics = df['zipcode'].apply(get_redis_data)
        df_demo = pd.DataFrame.from_records(demographics)

        if df_demo.empty:
            print("DEBUG (2. Redis): WARNING! df_demo is EMPTY. Redis lookup failed.")

        # 2. Combine with input data
        df_demo_clean = df_demo.drop('zipcode', axis=1, errors='ignore')
        df_combined = pd.concat([df.reset_index(drop=True), df_demo_clean.reset_index(drop=True)], axis=1)

        # 3. Fill NaNs
        df_combined = df_combined.fillna(0)

        print(f"DEBUG (3. Combined DF): Shape {df_combined.shape}")
        print(f"DEBUG (3. Combined DF): Columns: {list(df_combined.columns)}")

        # --- [ 4. THE FIX: Force Data Types ] ---
        # We must ensure dtypes match the training data
        print("DEBUG (4. Type Conversion): Forcing categorical dtypes...")
        try:
            # These were numbers in the source data
            numeric_categoricals = ['waterfront', 'view', 'condition', 'grade']
            for col in numeric_categoricals:
                if col in df_combined.columns:
                    df_combined[col] = df_combined[col].astype(int)

            # Zipcode was a string in the training data (dtype={'zipcode': str})
            if 'zipcode' in df_combined.columns:
                # Convert from number (e.g. 98178.0) -> int (98178) -> str ("98178")
                df_combined['zipcode'] = df_combined['zipcode'].astype(int).astype(str)

            print("DEBUG (4. Type Conversion): Dtypes forced successfully.")
        except Exception as e:
            print(f"DEBUG (4. Type Conversion): ERROR during type conversion: {e}")
            raise e
        # --- [ END OF FIX ] ---

        # 5. Predict
        print("DEBUG (5. Predict): Calling self.model.predict()...")
        try:
            prediction = self.model.predict(df_combined)
            print("--- [V3 Wrapper] PREDICT END (SUCCESS) ---")
            return prediction
        except KeyError as e:
            print("\n--- [V3 Wrapper] KEYERROR CAUGHT! ---")
            print(f"ERROR: {e}")
            raise e
        except Exception as e:
            print("\n--- [V3 Wrapper] UNEXPECTED ERROR CAUGHT! ---")
            print(f"ERROR: {e}")
            raise e


# --- CRITICAL LINE ---
print("Registering HousingModelWrapperV3 with mlflow.models.set_model()")
mlflow.models.set_model(HousingModelWrapperV3())