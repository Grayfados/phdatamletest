import pandas as pd
import numpy as np
import joblib
import json
import pathlib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

SALES_PATH = "../data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "../data/zipcode_demographics.csv"
OUTPUT_DIR = "model"
MODEL_V2_NAME = "housing-model-v2"
MLFLOW_EXPERIMENT_NAME = "phdata-housing-v2"


def engineer_features(df):
    """Creates new time-based features from the date column."""
    print("Starting feature engineering...")
    df_fe = df.copy()

    df_fe['date'] = pd.to_datetime(df_fe['date'].str.split('T').str[0], format='%Y%m%d')

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

    print("Feature engineering complete.")
    return df_fe


def load_data(sales_path, demographics_path):
    """Loads, merges, and cleans data for V2 model."""
    print("Loading data...")
    df_sales = pd.read_csv(sales_path, dtype={'zipcode': str})
    df_demo = pd.read_csv(demographics_path, dtype={'zipcode': str})

    df_merged = pd.merge(df_sales, df_demo, on='zipcode', how='left')

    y = df_merged.pop('price')
    X = df_merged

    return X, y


def build_pipeline(X_train):
    """Builds a ColumnTransformer and Pipeline for the V2 model."""
    print("Building preprocessing pipeline...")

    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    numeric_features.pop('lat')
    numeric_features.pop('lon')
    categorical_features = ['zipcode', 'waterfront', 'view', 'condition', 'grade']

    numeric_features = [col for col in numeric_features if col not in categorical_features]

    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep any columns not specified
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    return model_pipeline, (numeric_features + categorical_features)


def main():
    """Main training and MLflow logging function."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        mlflow.log_param("model_type", "RandomForestRegressor_v2")

        X, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH)
        X = engineer_features(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        model_pipeline, feature_list = build_pipeline(X_train)

        print("Starting model training...")
        model_pipeline.fit(X_train, y_train)
        print("Training complete.")

        print("Evaluating model...")
        y_pred = model_pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"--- V2 Model Metrics ---")
        print(f"  R²:   {r2:.4f}")
        print(f"  RMSE: ${rmse:,.2f}")

        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)

        print(f"Logging and registering model as '{MODEL_V2_NAME}'...")
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            name="model",
            registered_model_name=MODEL_V2_NAME
        )

        output_dir = pathlib.Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        model_v2_path = output_dir / "model_v2.pkl"
        features_v2_path = output_dir / "model_features_v2.json"

        joblib.dump(model_pipeline, model_v2_path)

        json.dump(list(X_train.columns), open(features_v2_path, 'w'))

        mlflow.log_artifact(str(model_v2_path))
        mlflow.log_artifact(str(features_v2_path))

        print(f"V2 Model artifacts saved to {OUTPUT_DIR}/")
        print("--- Run Finished ---")


if __name__ == "__main__":
    main()