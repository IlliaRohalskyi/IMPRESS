"""
Module Test Model Trainer.

This module contains unit tests for the ModelTrainer class in the model_trainer module.
"""

from io import StringIO
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.components.model_trainer import ModelTrainer, TrainTestData


@pytest.fixture(name="mock_data")
def fixture_mock_data():
    """
    Fixture for mock TrainTestData object.

    Returns:
        TrainTestData: A mock TrainTestData object.
    """
    x_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 3)
    x_test = np.random.rand(20, 10)
    y_test = np.random.rand(20, 3)
    feature_names = [
        "feature_0",
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5",
        "feature_6",
        "feature_7",
        "feature_8",
        "feature_9",
    ]
    return TrainTestData(x_train, y_train, x_test, y_test, feature_names)


@pytest.fixture(name="mock_savefig")
def fixture_mock_savefig():
    """
    Fixture for mocking saving figures using pyplot.
    """
    with patch("matplotlib.pyplot.savefig"):
        yield


@pytest.fixture(name="mock_rf")
def fixture_mock_rf():
    """
    Fixture for mocking RandomForestRegressor.

    Yields:
        Mock: A mock RandomForestRegressor object.
    """
    with patch("sklearn.ensemble.RandomForestRegressor") as mock_rf:
        yield mock_rf


@pytest.fixture(name="model_trainer")
def fixture_model_trainer(mock_data):
    """
    Fixture for creating a ModelTrainer instance.

    Args:
        mock_data (TrainTestData): A mock TrainTestData object.

    Returns:
        ModelTrainer: A ModelTrainer instance with mock data.
    """
    with patch("src.components.model_trainer.load_pickle") as mock_scaler:
        mock_scaler.return_value.inverse_transform.return_value = np.array(
            [
                [0.5, 0.6, 0.7],
                [0.4, 0.3, 0.2],
            ]
        )
        model_trainer_obj = ModelTrainer(mock_data)
    return model_trainer_obj


def test_feature_importance_plot(model_trainer, mock_savefig):  # pylint: disable=W0613
    """
    Test feature_importance_plot method.

    Args:
        model_trainer (ModelTrainer): A ModelTrainer instance.
        mock_savefig (None): A mock for saving figures.
    """
    with patch("matplotlib.pyplot.savefig"):
        mock_model = Mock()
        mock_model.feature_importances_ = np.random.rand(10)
        mock_model_name = "mock_model"
        model_trainer.data.feature_names = [
            "feature_0",
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
            "feature_6",
            "feature_7",
            "feature_8",
            "feature_9",
        ]
        with patch("sys.stdout", new_callable=StringIO):
            model_trainer.feature_importance_plot(mock_model, mock_model_name)


def test_train_model(mock_rf, mock_data, model_trainer):  # pylint: disable=W0613
    """
    Test train_model method.

    Args:
        mock_rf (Mock): A mock RandomForestRegressor object.
        mock_data (TrainTestData): A mock TrainTestData object.
        model_trainer (ModelTrainer): A ModelTrainer instance.
    """
    mock_best_params = {"n_estimators": 100, "max_depth": 10}

    with patch(
        "src.components.model_trainer.RandomForestRegressor", return_value=mock_rf
    ) as mock_rf_constructor:
        model_trainer.train_model("rf", mock_best_params)

    mock_rf_constructor.assert_called_once_with(n_estimators=100, max_depth=10)
    mock_rf.fit.assert_called_once_with(
        model_trainer.data.x_train, model_trainer.data.y_train
    )

    mock_rf_constructor.assert_called_once_with(n_estimators=100, max_depth=10)
    mock_rf.fit.assert_called_once_with(
        model_trainer.data.x_train, model_trainer.data.y_train
    )
