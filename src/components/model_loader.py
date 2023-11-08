"""
Model Loader Script

This script provides functions to download machine learning models 
from the MLFlow model registry, save them as a dictionary,
and retrieve these models for predictions.
"""

import os
import shutil
import sys
from dataclasses import dataclass
from typing import Tuple, Union

import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import get_project_root, load_pickle, save_pickle


@dataclass
class ModelAndScalers:
    """
    This dataclass stores models and scalers associated with a specific model
    """

    model: Union[RandomForestRegressor, XGBRegressor]
    model_name: str
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler


def get_models(
    model_names: Tuple[str] = ("rf", "xgb"),
    scalers_path: str = os.path.join(get_project_root(), "ml_downloads"),
):
    """
    Load production models and scalers from the MLFlow model registry.
    Save scalers at scalers_path directory.

    !!!!!!!!!!!!!WARNING!!!!!!!!!!!!
    This function needs MLFlow credentials

    Args:
        model_names (Tuple[str]): Names of the MLFlow models to load.
        scalers_path (str): The path where the  scalers will be saved as a pickle file.
        If not provided, the default path is used.

    Returns:
        List: A list of ModelAndScalers objects which stores the loaded model
        and scalers associated to the model
    """
    try:
        if os.path.exists(scalers_path):
            shutil.rmtree(scalers_path)
        for model_name in model_names:
            latest_versions = mlflow.tracking.MlflowClient().get_latest_versions(
                name=model_name, stages=["Production"]
            )

            if latest_versions:
                latest_version = latest_versions[0]
                mlflow.artifacts.download_artifacts(
                    run_id=latest_version.run_id,
                    artifact_path="scalers/",
                    dst_path=os.path.join(scalers_path, f"{model_name}"),
                )

                model = (
                    mlflow.xgboost.load_model(
                        f"runs:/{latest_version.run_id}/models/{model_name}"
                    )
                    if model_name == "xgb"
                    else mlflow.sklearn.load_model(
                        f"runs:/{latest_version.run_id}/models/{model_name}"
                    )
                )

                feature_scaler_path = os.path.join(
                    scalers_path, model_name, "scalers", "feature_scaler.pkl"
                )
                target_scaler_path = os.path.join(
                    scalers_path, model_name, "scalers", "target_scaler.pkl"
                )

                feature_scaler = load_pickle(feature_scaler_path)

                target_scaler = load_pickle(target_scaler_path)

                model_and_scalers = ModelAndScalers(
                    model, model_name, feature_scaler, target_scaler
                )
                save_pickle(
                    model_and_scalers,
                    os.path.join(scalers_path, model_name, "model_and_scalers.pkl"),
                )

    except Exception as error_message:
        logging.error(f"Model loader failed: {error_message}")
        raise CustomException(error_message, sys) from error_message


if __name__ == "__main__":
    get_models()
