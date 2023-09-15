"""
Test Module for Model Loader Script

This module contains unit tests for the functions in the Model Loader script.
It tests the functionality of loading machine learning models and associated scalers
from the MLFlow model registry.
"""
import os
import shutil

import pytest

from src.components.model_loader import get_models
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


def test_get_models_empty(test_directory):
    """
    Tests the get_models function with an empty directory.

    Args:
        test_directory (str): The temporary test directory path created by the fixture.
    """

    models_and_scalers = get_models(scalers_path=test_directory)

    assert isinstance(models_and_scalers, list)
    assert len(models_and_scalers) != 0
    shutil.rmtree(test_directory)
