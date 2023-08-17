from io import StringIO
import numpy as np
import pytest
from unittest.mock import Mock, patch
from src.components.model_trainer import ModelTrainer

@pytest.fixture
def mock_data():
    x_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 3)
    x_test = np.random.rand(20, 10)
    y_test = np.random.rand(20, 3)
    feature_names = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9']
    return x_train, y_train, x_test, y_test, feature_names

@pytest.fixture
def mock_savefig():
    with patch('matplotlib.pyplot.savefig') as mock:
        yield mock

@pytest.fixture
def mock_rf():
    with patch('sklearn.ensemble.RandomForestRegressor') as mock_rf:
        yield mock_rf

@pytest.fixture
def mock_git_hash():
    with patch('src.components.model_trainer.ModelTrainer._get_git_hash', return_value="mock_git_hash"):
        yield

@pytest.fixture
def mock_start_run():
    with patch('mlflow.start_run'):
        yield

@pytest.fixture
def model_trainer():
    return ModelTrainer()

def test_feature_importance_plot(mock_savefig, model_trainer):
    mock_model = Mock()
    mock_model.feature_importances_ = np.random.rand(10)
    mock_model_name = "mock_model"
    model_trainer.feature_names = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9']
    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        model_trainer.feature_importance_plot(mock_model, mock_model_name)

def test_train_model(mock_rf, mock_data, model_trainer):
    x_train, y_train, _, _, _ = mock_data
    mock_best_params = {'n_estimators': 100, 'max_depth': 10}

    with patch('src.components.model_trainer.RandomForestRegressor', return_value=mock_rf) as mock_rf_constructor:
        model_trainer.train_model(x_train, y_train, 'rf', mock_best_params)

    mock_rf_constructor.assert_called_once_with(n_estimators=100, max_depth=10)
    mock_rf.fit.assert_called_once_with(x_train, y_train)

def test_initiate_model_training(model_trainer, mock_data):
    x_train, y_train, x_test, y_test, feature_names = mock_data

    with patch('mlflow.start_run'), \
         patch('mlflow.log_param'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_artifact'), \
         patch('mlflow.log_artifacts'), \
         patch('mlflow.log_metric'), \
         patch('mlflow.xgboost.log_model'), \
         patch('mlflow.sklearn.log_model'), \
         patch('mlflow.end_run'), \
         patch('src.components.model_trainer.load_pickle'), \
         patch('src.components.model_trainer.save_pickle'), \
         patch('sklearn.preprocessing.StandardScaler.inverse_transform', side_effect=lambda x: x):

        mock_study = Mock()
        mock_study.best_params = {'n_estimators': 100, 'max_depth': 10}
        mock_study.best_value = 0.123
        with patch('optuna.create_study', return_value=mock_study):

            with patch('src.components.model_trainer.ModelTrainer.train_model') as mock_train_model:
                mock_predictions = np.random.rand(len(x_test), len(y_test[0]))
                mock_model = Mock()
                mock_model.predict.return_value = mock_predictions
                mock_train_model.return_value = mock_model
                mock_model.feature_importances_ = np.random.rand(len(feature_names))

                with patch('sklearn.metrics.mean_absolute_error', side_effect=lambda x, y, **kwargs: np.abs(x - y).mean()):
                    model_trainer.initiate_model_training(x_train, y_train, x_test, y_test, feature_names=feature_names)
                    mock_train_model.assert_called()
