"""
This module implements ModelLoader class to load models from MLFlow
"""
import os
from dataclasses import dataclass
from typing import List

from src.utils import get_project_root

# from mlflow.tracking import MlflowClient


@dataclass
class LoaderConfig:
    """
    Class that defines the configuration for ModelLoader
    """

    model_names: List[str] = ["xgb", "rf"]
    artifacts_download_path: str = os.path.join(get_project_root(), "downloads")


class ModelLoader:
    """
    ModelLoader class that loads model from MLFlow
    """

    def __init__(self):
        """
        Initializes the ModelLoader class
        """
        return

    def initiate_model_loading(self):
        """
        Checks whether the downloaded version missmatches and downloads the model from MLFlow
        """
        return

    def load_and_fetch_model(self, model_name, model_version):
        """
        Loads the model from MLFlow and downloads it's artifacts
        """
        return model_name, model_version
