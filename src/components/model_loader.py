"""
Model Loader Script

This script provides functions to download machine learning models from the MLFlow model registry, save them
as a dictionary, and retrieve these models for predictions.

"""

import os
import shutil
import sys
from typing import Tuple

import mlflow

from src.exception import CustomException
from src.logger import logging
from src.utils import get_project_root, load_pickle


def save_models(
    model_names: Tuple[str] = ("xgb", "rf"),
    models_path: str = os.path.join(get_project_root(), "ml_downloads"),
    save: bool = False,
    load: bool = True,
):
    """
    Load production models and scalers from the MLFlow model registry and save them.
    Args:
        model_names (Tuple[str]): Names of the MLFlow models to load.
        models_path (str): The path where the loaded models and scalers will be saved as a pickle file. If not provided,
            the default path is used.

    Returns:
        None
    """
    try:
        mlflow.set_tracking_uri("https://dagshub.com/IlliaRohalskyi/IMPRESS.mlflow")
        if save:
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
                        artifact_path="scalers/",
                        dst_path=os.path.join(models_path, f"{model_name}"),
                    )

                    mlflow.artifacts.download_artifacts(
                        run_id=latest_version.run_id,
                        artifact_path=f"models/{model_name}",
                        dst_path=os.path.join(models_path, f"{model_name}"),
                    )
        if load:
            models_and_scalers = {}
            for model_name in model_names:
                model_path = os.path.join(
                    models_path, model_name, "models", "model.pkl"
                )
                model = load_pickle(model_path)

    except Exception as error_message:
        logging.error(f"Model loader failed: {error_message}")
        raise CustomException(error_message, sys) from error_message


if __name__ == "__main__":
    save_models(return_models=False)
