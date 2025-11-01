# File: register_wrapper.py (Corrected based on documentation)

import mlflow
import os

# --- Config ---
MLFLOW_EXPERIMENT_NAME = "phdata-housing-v2"
REGISTERED_MODEL_NAME = "housing-model-v2"
MODEL_V2_ARTIFACT_PATH = "model/model_v2.pkl"
WRAPPER_CODE_FILE = "housing_wrapper.py"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"Starting run {run_id} to register the wrapper (Model from Code method)...")

    # 1. Define artifacts the wrapper needs
    artifacts = {
        "model_pipeline": MODEL_V2_ARTIFACT_PATH
    }

    pip_requirements = [
        "mlflow",
        "scikit-learn==1.3.1",
        "pandas",
        "numpy",
        "redis"
    ]
    print("Logging model (Model from Code method)...")

    mlflow.pyfunc.log_model(
        artifact_path="model_wrapper",
        python_model=WRAPPER_CODE_FILE,
        code_paths=[WRAPPER_CODE_FILE],
        artifacts=artifacts,
        pip_requirements=pip_requirements,
        registered_model_name=REGISTERED_MODEL_NAME
    )

    mlflow.log_param("model_type", "wrapper_v2_model_from_code")
    print(f"\nWrapper model '{REGISTERED_MODEL_NAME}' logged and registered successfully.")
    print("This should resolve the 'model_class' TypeError.")