"""
Model Loader Script

This script provides functions to download machine learning models from the MLFlow model registry, save them
as a dictionary, and retrieve these models for predictions.

"""

import os
import shutil
from typing import Tuple

import mlflow

from src.utils import get_project_root


def load_and_save_models(
    model_names: Tuple[str] = ("xgb", "rf"),
    models_path: str = os.path.join(get_project_root(), "ml_downloads"),
):
    """
    Load production models and scalers from the MLFlow model registry, save them as a dictionary, and optionally
    specify the path to save the models.

    Args:
        model_names (Tuple[str]): Names of the MLFlow models to load.
        models_path (str): The path where the loaded models and scalers will be saved as a pickle file. If not provided,
            the default path is used.

    Returns:
        None
    """

    mlflow.set_tracking_uri("https://dagshub.com/IlliaRohalskyi/IMPRESS.mlflow")
    if os.path.exists(models_path):
        shutil.rmtree(models_path)

    for model_name in model_names:
        latest_versions = mlflow.tracking.MlflowClient().get_latest_versions(
            name=model_name, stages=["Production"]
        )

        if latest_versions:
            latest_version = latest_versions[0]
            mlflow.artifacts.download_artifacts(
                run_id=latest_version.run_id,
                artifact_path="scalers/feature_scaler.pkl",
                dst_path=os.path.join(models_path, f"{model_name}"),
            )

            mlflow.artifacts.download_artifacts(
                run_id=latest_version.run_id,
                artifact_path=f"models/{model_name}",
                dst_path=os.path.join(models_path, f"{model_name}"),
            )


if __name__ == "__main__":
    load_and_save_models()
