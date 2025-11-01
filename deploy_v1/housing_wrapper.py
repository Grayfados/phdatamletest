# File: housing_wrapper.py
# (Code in English, as requested)

import mlflow
import joblib
import redis
import json
import pandas as pd
import numpy as np


# --- Feature Engineering Function ---
def engineer_features(df):
    """Creates new time-based features from the date column."""
    df_fe = df.copy()

    try:
        df_fe['date'] = pd.to_datetime(df_fe['date'].str.split('T').str[0], format='%Y%m%d')
    except Exception as e:
        print(f"Warning: Could not parse all dates. Error: {e}")
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

    # --- NO __init__ method! ---

    def load_context(self, context):
        """
        Called when loading the model for serving, NOT when logging.
        """
        print("Loading model from context...")
        self.model = joblib.load(context.artifacts["model_pipeline"])

        print("Connecting to Redis (localhost:6379)...")
        try:
            # Assuming Redis is exposed on localhost for the *serving* environment
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            print("Redis connection successful.")
        except redis.exceptions.ConnectionError as e:
            print(f"FATAL: Error connecting to Redis: {e}")
            self.redis_client = None

    def predict(self, context, model_input):
        """
        Called for inference.
        """
        if self.redis_client is None:
            raise Exception("Redis connection not available. Service is unhealthy.")

        df = model_input.copy()

        def get_redis_data(zipcode):
            try:
                clean_zip = str(int(float(zipcode)))
                data_json = self.redis_client.get(clean_zip)
                if data_json:
                    return json.loads(data_json)
            except Exception:
                pass
            return {}

        print(f"Fetching demographics for {len(df)} zipcodes...")
        demographics = df['zipcode'].apply(get_redis_data)
        df_demo = pd.DataFrame.from_records(demographics)

        df_demo = df_demo.drop('zipcode', axis=1, errors='ignore')
        df_combined = pd.concat([df.reset_index(drop=True), df_demo.reset_index(drop=True)], axis=1)

        print("Performing feature engineering...")
        df_final_features = engineer_features(df_combined)

        print("Calling internal model pipeline for prediction...")
        return self.model.predict(df_final_features)


# --- THIS IS THE FIX ---
# Add this line at the end of the file (at the module level)
# This tells MLflow which object IS the model
print("Registering HousingModelWrapper with mlflow.models.set_model()")
mlflow.models.set_model(HousingModelWrapper())
# --- END OF FIX ---