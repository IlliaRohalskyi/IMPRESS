"""
Prefect Flow for End-to-End Training Pipeline

This script defines a Prefect flow that orchestrates the execution of data ingestion, 
transformation, and model training tasks
"""
import os

from prefect import flow, task

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import get_project_root


@task
def data_ingestion():
    """
    Task for data ingestion from online and offline sources.

    Returns:
        dict: Dictionary containing online and offline data.
    """
    online_data, offline_data = DataIngestion().initiate_data_ingestion()
    return {"online_data": online_data, "offline_data": offline_data}


@task
def data_transformation(result):
    """
    Task for data transformation.

    Args:
        result (dict): Dictionary containing online and offline data.

    Returns:
        dict: Dictionary containing transformed train-test data.
    """
    online_data = result["online_data"]
    offline_data = result["offline_data"]

    washing, rinsing = DataTransformation().initiate_data_transformation(
        online_data, offline_data
    )
    return {"washing": washing, "rinsing": rinsing}


@task
def model_training(result):
    """
    Task for initiating model training using transformed data.

    Args:
        result (dict): Dictionary containing transformed train-test data.
    """
    washing = result["washing"]
    rinsing = result["rinsing"]

    washing_trainer = ModelTrainer(washing)
    rinsing_trainer = ModelTrainer(rinsing)

    washing_trainer.trainer_paths.feature_scaler_path = os.path.join(
        get_project_root(), "artifacts/data_processing/washing_feature_scaler.pkl"
    )
    washing_trainer.trainer_paths.target_scaler_path = os.path.join(
        get_project_root(), "artifacts/data_processing/washing_feature_scaler.pkl"
    )

    rinsing_trainer.trainer_paths.feature_scaler_path = os.path.join(
        get_project_root(), "artifacts/data_processing/rinsing_feature_scaler.pkl"
    )
    rinsing_trainer.trainer_paths.target_scaler_path = os.path.join(
        get_project_root(), "artifacts/data_processing/rinsing_feature_scaler.pkl"
    )

    washing_trainer.initiate_model_training()
    rinsing_trainer.initiate_model_training()


@flow(name="train_pipeline")
def train_pipeline():
    """
    Prefect flow that orchestrates the execution of data ingestion, transformation,
    and model training tasks.
    """
    result_ingestion = data_ingestion()
    result_transformation = data_transformation(result=result_ingestion)
    model_training(result=result_transformation)


if __name__ == "__main__":
    train_pipeline()
