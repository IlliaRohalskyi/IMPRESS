"""
Predict Pipeline Integration Tests

This module contains integration tests for the prediction pipeline components.
It tests the interaction between the data transformation and ensemble prediction tasks
to ensure that the pipeline generates predictions correctly.
"""
import os
import shutil

import pandas as pd
import pytest

from src.components.model_loader import get_models
from src.pipelines.predict_pipeline import (data_transformation,
                                            ensemble_predict)
from src.utils import get_project_root


@pytest.fixture(name="test_directory")
def fixture_test_directory():
    """
    Creates a temporary test directory for testing.

    Returns:
        str: The path to the temporary test directory.
    """
    test_dir = os.path.join(
        get_project_root(), "tests", "test_data", "test_model_loader"
    )
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir


@pytest.fixture(name="pred_data")
def fixture_pred_data():
    """
    Loads online synthetic data and formats it as expected for prediction

    Returns:
        pd.DataFrame: Dataframe that is ready to be fed into prediction pipeline
    """
    synthetic_data = pd.read_csv(
        os.path.join(get_project_root(), "tests", "test_data", "synthetic_online.csv"),
        sep=";",
    )
    synthetic_data["waschen"] = 1
    synthetic_pred = synthetic_data[
        [
            "waschen",
            "truebung",
            "druck1",
            "fluss1",
            "ph",
            "leitfaehigkeit",
        ]
    ]
    yield synthetic_pred


def test_get_models_empty(test_directory, pred_data):
    """
    Tests interaction between prediction pipeline components. Simulates the flow

    Args:
        test_directory (str): The temporary test directory path created by the fixture.
        pred_data (pd.DataFrame): The dataframe to run pipeline on.
    """

    models_and_scalers = get_models(scalers_path=test_directory)

    result_transformation = data_transformation.fn(
        pred_data=pred_data, models_and_scalers=models_and_scalers
    )
    predictions = ensemble_predict.fn(
        result=result_transformation, models_and_scalers=models_and_scalers
    )
    print(predictions.shape)
    assert predictions.shape[1] == 3

    shutil.rmtree(test_directory)
