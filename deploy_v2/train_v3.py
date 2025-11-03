import pandas as pd
import numpy as np
import joblib
import json
import pathlib
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import randint

SALES_PATH = os.path.join("..", "data", "kc_house_data.csv")
DEMOGRAPHICS_PATH = os.path.join("..", "data", "zipcode_demographics.csv")
OUTPUT_DIR = "model"
MODEL_V3_NAME = "housing-model-v3-production"
MLFLOW_EXPERIMENT_NAME = "phdata-housing-final"


def load_data(sales_path, demographics_path):
    print("Loading data...")
    df_sales = pd.read_csv(sales_path, dtype={'zipcode': str})
    df_demo = pd.read_csv(demographics_path, dtype={'zipcode': str})

    df_merged = pd.merge(df_sales, df_demo, on='zipcode', how='left')

    y = df_merged.pop('price')
    X = df_merged

    X = X.drop(['id', 'date'], axis=1, errors='ignore')

    return X, y


def build_pipeline(X_train):
    print("Building preprocessing pipeline...")

    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = ['zipcode', 'waterfront', 'view', 'condition', 'grade']

    numeric_features = [col for col in numeric_features if col not in categorical_features]

    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    return model_pipeline


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Hyperparameter Tuning") as parent_run:
        print(f"MLflow Parent Run ID: {parent_run.info.run_id}")

        X, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH)
        X = X.fillna(0)
        y = y.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_pipeline = build_pipeline(X_train)

        param_dist = {
            'regressor__n_estimators': randint(50, 500),
            'regressor__max_depth': randint(10, 100),
            'regressor__min_samples_leaf': randint(1, 10),
            'regressor__min_samples_split': randint(2, 20)
        }

        n_iter_search = 10
        cv_folds = 5

        search = RandomizedSearchCV(
            estimator=model_pipeline,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            cv=cv_folds,
            scoring='r2',
            random_state=42,
            n_jobs=-1
        )

        print(f"Starting RandomizedSearchCV (n_iter={n_iter_search}, cv={cv_folds})...")
        search.fit(X_train, y_train)
        print("Tuning complete.")

        print(f"Logging {n_iter_search} child runs for comparison...")
        cv_results = search.cv_results_

        for i in range(n_iter_search):
            with mlflow.start_run(run_name=f"trial_{i}", nested=True) as child_run:
                params = cv_results['params'][i]
                mlflow.log_params(params)

                score = cv_results['mean_test_score'][i]
                mlflow.log_metric("cv_r2", score)

        print("Child runs logged.")

        mlflow.log_param("tuning_n_iter", n_iter_search)
        mlflow.log_metric("best_cv_r2", search.best_score_)
        mlflow.log_params(search.best_params_)

        best_model = search.best_estimator_

        print("Evaluating best model on test set...")
        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"--- Tuned V3 Model Test Metrics ---")
        print(f"  Test R²:   {r2:.4f}")
        print(f"  Test RMSE: ${rmse:,.2f}")

        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("test_rmse", rmse)

        output_dir = pathlib.Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        model_v3_path = output_dir / "model_v3.pkl"
        joblib.dump(best_model, model_v3_path)
        print(f"Tuned V3 Model artifact saved to {model_v3_path}")

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model_artifact"
        )
        print("Best model artifact logged to parent run.")


if __name__ == "__main__":
    main()