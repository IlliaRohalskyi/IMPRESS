"""
Prediction Pipeline Prefect Flow

This module defines a Prefect flow for data transformation and ensemble model prediction.

Tasks:
1. data_transformation: Task for transforming input data for each model.
2. ensemble_predict: Task for making predictions using an ensemble of models.

The flow orchestrates the execution of these tasks to generate predictions.
"""
from prefect import flow, task

from src.components.data_transformation import DataTransformation


@task
def data_transformation(pred_data, models_and_scalers):
    """
    Task for data transformation.

    Args:
        result (dict): Dictionary containing scaled online data for each model.

    Returns:
        dict: Dictionary containing transformed train-test data.
    """
    return_dict = {}
    tr_data = DataTransformation().initiate_data_transformation(pred_data, None)
    for model_and_scalers in models_and_scalers:
        scaled_data = model_and_scalers.feature_scaler.transform(tr_data)
        return_dict[f"transformed_data_{model_and_scalers.model_name}"] = scaled_data
    return return_dict


@task
def ensemble_predict(result, models_and_scalers):
    """
    Task for initiating model prediction using transformed data.

    Args:
        result (dict): Dictionary containing transformed prediction data.

    Returns:
        np.array: Array containing predictions of the model.
    """

    weights = {"xgb": 0.49, "rf": 0.51}
    preds_scaled = 0
    for model_and_scalers in models_and_scalers:
        tr_data = result[f"transformed_data_{model_and_scalers.model_name}"]
        preds = (
            model_and_scalers.model.predict(tr_data)
            * weights[model_and_scalers.model_name]
        )
        preds_scaled += model_and_scalers.target_scaler.inverse_transform(preds)

    return preds_scaled


@flow(name="prediction_pipeline")
def prediction_pipeline(models_and_scalers, pred_data):
    """
    Prefect flow that orchestrates the execution of data transformation,
    and model prediction tasks.
    """
    result_transformation = data_transformation(
        pred_data=pred_data, models_and_scalers=models_and_scalers
    )
    predictions = ensemble_predict(
        result=result_transformation, models_and_scalers=models_and_scalers
    )
    return predictions
