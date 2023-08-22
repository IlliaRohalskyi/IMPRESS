"""
Prefect Flow for End-to-End Retraining Pipeline

This script defines a Prefect flow that orchestrates the execution of data ingestion, 
transformation, and model retraining tasks
"""
from dataclasses import dataclass

from prefect import flow, task

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class ModelParams:
    """
    This class stores all the parameters necessary to retrain the model

    Args:
        model_name: The name of the model. "xgb" for xgboost, "rf" for RandomForest
        params: The model parameters depending on the model_name
    """

    model_name: str = "rf"
    if model_name == "xgb":
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "min_child_weight": 1,
            "reg_alpha": 0,
            "reg_lambda": 1,
        }
    elif model_name == "rf":
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 0.7,
            "bootstrap": True,
        }


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
    model_params = ModelParams()
    ModelTrainer(train_test_data_obj).retrain_model(
        model_params.model_name, model_params.params
    )


@flow(name="retrain_pipeline")
def retrain_pipeline():
    """
    Prefect flow that orchestrates the execution of data ingestion, transformation,
    and model training tasks.
    """
    result_ingestion = data_ingestion()
    result_transformation = data_transformation(result=result_ingestion)
    model_training(result=result_transformation)


if __name__ == "__main__":
    retrain_pipeline()
