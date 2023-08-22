"""
Prefect Flow for End-to-End Training Pipeline

This script defines a Prefect flow that orchestrates the execution of data ingestion, 
transformation, and model training tasks
"""

from prefect import flow, task

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


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

    train_test_data = DataTransformation().initiate_data_transformation(
        online_data, offline_data
    )
    return {"train_test_data": train_test_data}


@task
def model_training(result):
    """
    Task for initiating model training using transformed data.

    Args:
        result (dict): Dictionary containing transformed train-test data.
    """
    train_test_data_obj = result["train_test_data"]
    ModelTrainer(train_test_data_obj).initiate_model_training()


@flow
def main_flow():
    """
    Prefect flow that orchestrates the execution of data ingestion, transformation,
    and model training tasks.
    """
    result_ingestion = data_ingestion()
    result_transformation = data_transformation(result=result_ingestion)
    model_training(result=result_transformation)


if __name__ == "__main__":
    main_flow()
