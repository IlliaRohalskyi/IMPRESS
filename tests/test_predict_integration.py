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
                                            ensemble_predict,
                                            load_data_and_models)
from src.utils import get_project_root


@pytest.fixture(name="test_directory")
def fixture_test_directory():
    """
    Creates a temporary test directory for testing.

    Returns:
        str: The path to the temporary test directory.
    """
    test_dir = os.path.join(
        get_project_root(), "tests", "test_data", "test_predict_integration"
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
    synthetic_data.drop(columns=["experimentnummer"])

    yield synthetic_data


def test_integration(test_directory, pred_data):
    """
    Tests interaction between prediction pipeline components. Simulates the flow

    Args:
        test_directory (str): The temporary test directory path created by the fixture.
        pred_data (pd.DataFrame): The dataframe to run pipeline on.
    """

    get_models(scalers_path=test_directory)

    result_load = load_data_and_models.fn()
    result_transformation = data_transformation.fn(result_load)
    predictions = ensemble_predict.fn(result_transformation)
    assert predictions.shape[1] == 3

    shutil.rmtree(test_directory)
