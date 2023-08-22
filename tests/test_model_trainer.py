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
    return ModelTrainer(mock_data)


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


def test_initiate_model_training(model_trainer, mock_savefig):  # pylint: disable=W0613
    """
    Test initiate_model_training method.

    Args:
        model_trainer (ModelTrainer): A ModelTrainer instance.
        mock_savefig (None): A mock for saving figures.
    """
    mock_study = Mock()
    mock_study.best_params = {"n_estimators": 100, "max_depth": 10}
    mock_study.best_value = 0.123
    with patch("optuna.create_study", return_value=mock_study), patch.object(
        model_trainer, "train_model"
    ) as mock_train_model, patch(
        "sklearn.metrics.mean_absolute_error",
        side_effect=lambda x, y, **kwargs: np.abs(x - y).mean(),
    ):
        mock_predictions = np.random.rand(
            len(model_trainer.data.x_test), len(model_trainer.data.y_test[0])
        )
        mock_model = Mock()
        mock_model.predict.return_value = mock_predictions
        mock_train_model.return_value = mock_model
        mock_model.feature_importances_ = np.random.rand(
            len(model_trainer.data.feature_names)
        )

        with patch("mlflow.start_run"), patch("mlflow.log_param"), patch(
            "mlflow.log_params"
        ), patch("mlflow.log_artifact"), patch("mlflow.log_artifacts"), patch(
            "mlflow.log_metric"
        ), patch(
            "mlflow.xgboost.log_model"
        ), patch(
            "mlflow.sklearn.log_model"
        ), patch(
            "mlflow.end_run"
        ), patch(
            "src.components.model_trainer.load_pickle"
        ), patch(
            "sklearn.preprocessing.StandardScaler.inverse_transform",
            side_effect=lambda x: x,
        ):
            model_trainer.initiate_model_training()

        mock_train_model.assert_called()


def test_retrain_model(model_trainer):
    """
    Test the retrain_model method.

    Args:
        model_trainer (ModelTrainer): A ModelTrainer instance.
    """
    mock_best_params = {"n_estimators": 100, "max_depth": 10}
    mock_model_name = "rf"
    mock_predictions = np.random.rand(
        len(model_trainer.data.x_test), len(model_trainer.data.y_test[0])
    )
    mock_model = Mock()
    mock_model.predict.return_value = mock_predictions
    model_trainer.train_model = Mock(return_value=mock_model)

    with patch("mlflow.set_tracking_uri"), patch("mlflow.start_run"), patch(
        "mlflow.xgboost.log_model"
    ), patch("mlflow.sklearn.log_model"), patch("mlflow.log_metric"), patch(
        "mlflow.log_artifacts"
    ), patch(
        "mlflow.log_param"
    ), patch(
        "mlflow.log_artifact"
    ), patch(
        "src.components.model_trainer.load_pickle"
    ), patch(
        "sklearn.preprocessing.StandardScaler.inverse_transform",
        side_effect=lambda x: x,
    ), patch(
        "src.components.model_trainer.ModelTrainer.feature_importance_plot"
    ):
        model_trainer.retrain_model(mock_model_name, mock_best_params)

    model_trainer.train_model.assert_called_once_with(mock_model_name, mock_best_params)
    mock_model.predict.assert_called_once_with(model_trainer.data.x_test)
