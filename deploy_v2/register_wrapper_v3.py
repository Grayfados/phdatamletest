import mlflow
import os

MLFLOW_EXPERIMENT_NAME = "phdata-housing-final"
REGISTERED_MODEL_NAME = "housing-model-v3-production"
MODEL_V3_ARTIFACT_PATH = "model/model_v3.pkl"
WRAPPER_CODE_FILE = "housing_wrapper_v3.py"

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run() as run:
    print(f"Starting run {run.info.run_id} to register the V3 wrapper...")

    artifacts = {
        "model_v3_pipeline": MODEL_V3_ARTIFACT_PATH
    }

    pip_requirements = [
        "mlflow",
        "scikit-learn==1.3.1",
        "pandas",
        "numpy",
        "redis"
    ]

    print("Logging V3 model (Model from Code method)...")

    mlflow.pyfunc.log_model(
        artifact_path="model_wrapper_v3",
        python_model=WRAPPER_CODE_FILE,
        code_paths=[WRAPPER_CODE_FILE],
        artifacts=artifacts,
        pip_requirements=pip_requirements,
        registered_model_name=REGISTERED_MODEL_NAME
    )

    print(f"\nV3 Wrapper model '{REGISTERED_MODEL_NAME}' logged and registered.")