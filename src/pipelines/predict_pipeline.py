"""
Prediction Pipeline Prefect Flow

This module defines a Prefect flow for data transformation and ensemble model prediction.

Tasks:
1. load_data_and_models: Task for loading machine learning models and prediction data.
2. data_transformation: Task for transforming input data for each model.
3. ensemble_predict: Task for making predictions using an ensemble of models.
4. write_and_delete_data: Task for writing and deleting data.

The flow orchestrates the execution of these tasks to generate predictions.
"""

import os
import sys

import pandas as pd
import psycopg2
from prefect import flow, task
from sqlalchemy import create_engine

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.utils import get_project_root, load_pickle


@task
def load_data_and_models(
    table_name, ml_downloads_path=os.path.join(get_project_root(), "ml_downloads")
):
    """
    Load Machine Learning Models and Prediction Data.

    Args:
        table_name: Name of the table to predict in PostgreSQL database.
        ml_downloads_path (str): Path to the directory containing downloaded models.
                                          Defaults to 'ml_downloads' in the project root.

    Returns:
        dict: A dictionary containing loaded models and prediction data.
    """
    try:
        models_and_scalers = []
        for model_name in ["xgb", "rf"]:
            model_and_scalers = load_pickle(
                os.path.join(ml_downloads_path, model_name, "model_and_scalers.pkl")
            )
            models_and_scalers.append(model_and_scalers)
        pred_data = DataIngestion().get_sql_table(table_name)
        if len(pred_data) == 0:
            return {}
        return {"models_and_scalers": models_and_scalers, "pred_data": pred_data}
    except Exception as error_message:
        logging.error(f"Error loading data and models: {error_message}")
        raise CustomException(error_message, sys) from error_message


@task
def data_transformation(results):
    """
    Task for data transformation.

    Args:
        results (dict): A dictionary containing ModelsAndScalers and prediction data.

    Returns:
        dict: A dictionary containing transformed prediction data.
    """
    try:
        pred_data = results["pred_data"]
        models_and_scalers = results["models_and_scalers"]

        return_dict = {}
        tr_data = DataTransformation().initiate_data_transformation(pred_data, None)

        return_dict["tr_data"] = tr_data

        for model_and_scalers in models_and_scalers:
            scaler = model_and_scalers.feature_scaler
            scaler_columns = scaler.feature_names_in_

            scaled_data = tr_data[scaler_columns]

            scaled_data = scaler.transform(scaled_data)

            return_dict[
                f"transformed_data_{model_and_scalers.model_name}"
            ] = scaled_data
        return_dict["models_and_scalers"] = models_and_scalers
        return return_dict
    except Exception as error_message:
        logging.error(f"Error during data transformation: {error_message}")
        raise CustomException(error_message, sys) from error_message


@task
def ensemble_predict(result):
    """
    Task for initiating model prediction using transformed data.

    Args:
        result (dict): A dictionary containing transformed prediction data.

    Returns:
        dict: modified dictionary containing predictions
    """
    try:
        weights = {"xgb": 0.49, "rf": 0.51}
        models_and_scalers = result["models_and_scalers"]
        output = {}
        preds_scaled = 0
        for model_and_scalers in models_and_scalers:
            tr_data = result[f"transformed_data_{model_and_scalers.model_name}"]
            preds = (
                model_and_scalers.model.predict(tr_data)
                * weights[model_and_scalers.model_name]
            )
            preds_scaled += model_and_scalers.target_scaler.inverse_transform(preds)
        output["tr_data"] = result["tr_data"]
        output["preds"] = preds_scaled
        return output
    except Exception as error_message:
        logging.error(f"Error during ensemble prediction: {error_message}")
        raise CustomException(error_message, sys) from error_message


@task
def write_and_delete_data(results, table_name, write_table_name):
    """
    Write predictions to archived table and delete source data.

    Args:
        results (dict): A dictionary containing predictions and transformed data.
        table_name (str): The name of the table to delete after the prediction.
        write_table_name (str): The name of the table to write the predictions to.
    """
    try:
        hostname = os.environ.get("DB_HOSTNAME")
        database_name = os.environ.get("DB_NAME")
        username = os.environ.get("DB_USERNAME")
        password = os.environ.get("DB_PASSWORD")

        preds = results["preds"]
        tr_data = results["tr_data"]

        preds_df = pd.DataFrame(
            preds,
            columns=[
                "oberflaechenspannung",
                "anionischetenside",
                "nichtionischentenside",
            ],
        )

        dataframe = pd.concat(
            [
                tr_data[["experimentnummer", "waschen"]],
                preds_df,
                tr_data.drop(["experimentnummer", "waschen"], axis=1),
            ],
            axis=1,
        )

        connection = psycopg2.connect(
            host=hostname, database=database_name, user=username, password=password
        )

        cursor = connection.cursor()

        engine = create_engine(
            f"postgresql://{username}:{password}@{hostname}/{database_name}"
        )

        dataframe.to_sql(write_table_name, engine, if_exists="append", index=False)

        delete_query = f"DELETE FROM {table_name}"
        cursor.execute(delete_query)
        connection.commit()

        cursor.close()
        connection.close()
    except Exception as error_message:
        logging.error(f"Error during data write and delete: {error_message}")
        raise CustomException(error_message, sys) from error_message


@flow(name="prediction_pipeline")
def prediction_pipeline(
    pred_table_name="online_data", write_table_name="archived_data"
):
    """
    Prefect flow that orchestrates the execution of data loading,
    data transformation, and model prediction tasks.

    Args:
        pred_table_name: The name of the prediction table in PostgreSQL database.
        write_table_name: The name of the table to write the predictions to.
    """
    result_load = load_data_and_models(pred_table_name)
    if result_load:
        result_transformation = data_transformation(result_load)
        result_predict = ensemble_predict(result_transformation)
        write_and_delete_data(result_predict, pred_table_name, write_table_name)


if __name__ == "__main__":
    prediction_pipeline()
