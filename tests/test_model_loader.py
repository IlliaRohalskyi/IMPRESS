import os
import shutil

import pytest

from src.components.model_loader import get_models
from src.utils import get_project_root


@pytest.fixture(scope="function")
def test_directory():
    test_dir = os.path.join(
        get_project_root(), "tests", "test_data", "test_model_loader"
    )
    os.makedirs(test_dir)
    yield test_dir


def test_get_models_empty(test_directory):
    # Call the function with the test directory as scalers path
    models_and_scalers = get_models(scalers_path=test_directory)

    # Add your assertions here
    assert isinstance(models_and_scalers, list)
    assert len(models_and_scalers) != 0
    shutil.rmtree(test_directory)
