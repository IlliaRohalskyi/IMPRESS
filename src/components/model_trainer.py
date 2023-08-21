"""
Model Trainer Module

This module contains the implementation of a class, ModelTrainer, which is responsible for
training and evaluating machine learning models using Optuna for hyperparameter
optimization and MLflow for tracking experiments.

Classes:
    ModelTrainer: A class for training and evaluating machine learning models.

Dataclasses:
    ModelTrainerPaths: Dataclass containing paths for various components used by ModelTrainer.
    TrainTestData: Dataclass representing training and testing data along with feature names.

Usage:
    Instantiate the ModelTrainer class with TrainTestData, call initiate_model_training()
    to train models using hyperparameter optimization, and log results and artifacts to MLflow.
"""
import os
import subprocess
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.components.data_transformation import TrainTestData
from src.exception import CustomException
from src.logger import logging
from src.utils import get_project_root, load_pickle


@dataclass
class ModelTrainerPaths:
    """
    Paths for various components used by ModelTrainer.
    """

    data_ingestion_script_path = os.path.join(
        get_project_root(), "src/components/data_ingestion.py"
    )

    data_transformation_script_path = os.path.join(
        get_project_root(), "src/components/data_transformation.py"
    )

    feature_scaler_path = os.path.join(
        get_project_root(), "artifacts/data_processing/feature_scaler.pkl"
    )

    target_scaler_path = os.path.join(
        get_project_root(), "artifacts/data_processing/target_scaler.pkl"
    )

    explainability_path = os.path.join(get_project_root(), "artifacts/explainability")


class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    """

    def __init__(self, train_test_data: TrainTestData):
        """
        Initialize the ModelTrainer.

        Args:
            train_test_data (TrainTestData): Data for training and testing.
        """
        self.data = train_test_data
        self.target_scaler = None
        self.current_model_name = None
        self.git_hash = self._get_git_hash()
        self.trainer_paths = ModelTrainerPaths()

    def train_model(self, model_name, best_params):
        """
        Train a machine learning model.

        Args:
            model_name (str): Name of the model.
            best_params (dict): Best hyperparameters for the model.

        Returns:
            model: Trained machine learning model.
        """
        try:
            logging.info("Training model")

            if model_name == "rf":
                model = RandomForestRegressor(**best_params)
            elif model_name == "xgb":
                model = XGBRegressor(tree_method="gpu_hist", **best_params)
            else:
                raise ValueError(f"Unsupported model name: {model_name}")

            model.fit(self.data.x_train, self.data.y_train)

        except Exception as error_message:
            logging.error(f"Model training failed with error: {error_message}")
            raise CustomException(error_message, sys) from error_message

        return model

    def objective(self, trial):
        """
        Objective function for Optuna hyperparameter optimization.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: Mean absolute error (MAE) for the model.
        """
        model_name = self.current_model_name

        if model_name == "xgb":
            model = XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 5, 30000),
                max_depth=trial.suggest_int("max_depth", 3, 30000),
                learning_rate=trial.suggest_float("learning_rate", 0.001, 0.3),
                subsample=trial.suggest_float("subsample", 0.1, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
                gamma=trial.suggest_float("gamma", 0, 10),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                reg_alpha=trial.suggest_float("reg_alpha", 0, 1),
                reg_lambda=trial.suggest_float("reg_lambda", 0, 1),
            )
        elif model_name == "rf":
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 5, 30000),
                max_depth=trial.suggest_int("max_depth", 3, 30000),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 30),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 15),
                max_features=trial.suggest_float("max_features", 0.1, 1),
                bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            )

        mae_list = []
        k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in k_fold.split(self.data.x_train, self.data.y_train):
            x_train_fold, x_val_fold = (
                self.data.x_train[train_idx],
                self.data.x_train[val_idx],
            )
            y_train_fold, y_val_fold = (
                self.data.y_train[train_idx],
                self.data.y_train[val_idx],
            )

            model.fit(x_train_fold, y_train_fold)
            predictions = model.predict(x_val_fold)

            mae = self.target_scaler.inverse_transform(
                mean_absolute_error(
                    y_val_fold, predictions, multioutput="raw_values"
                ).reshape(1, -1)
            )

            average_scaled_mae = np.mean(mae)

            mae_list.append(average_scaled_mae)

        return np.mean(mae_list)

    def _get_git_hash(self):
        """
        Get the Git hash of the current commit.

        Returns:
            str: Git hash.
        """
        try:
            logging.info("Getting git hash")

            git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .strip()
                .decode("utf-8")
            )
            return git_hash
        except subprocess.CalledProcessError as error_message:
            logging.error(
                f"Error: Failed to retrieve Git hash. "
                f"Command returned {error_message.returncode}: {error_message.output}"
            )
            return "Unknown"

    def initiate_model_training(self):
        """
        Initialize the model training process.
        """
        try:
            logging.info("Starting model training")

            self.target_scaler = load_pickle(self.trainer_paths.target_scaler_path)

            models = ["xgb", "rf"]

            mlflow.set_tracking_uri("https://dagshub.com/IlliaRohalskyi/IMPRESS.mlflow")

            with mlflow.start_run():
                mlflow.log_param("git_hash", self.git_hash)

                mlflow.log_artifact(
                    self.trainer_paths.feature_scaler_path, artifact_path="scalers"
                )

                mlflow.log_artifact(
                    self.trainer_paths.target_scaler_path, artifact_path="scalers"
                )

                mlflow.log_artifact(
                    self.trainer_paths.data_ingestion_script_path,
                    artifact_path="components",
                )

                mlflow.log_artifact(
                    self.trainer_paths.data_transformation_script_path,
                    artifact_path="components",
                )

                best_models = []
                best_maes = []

                for model_name in models:
                    self.current_model_name = model_name

                    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=35)
                    study = optuna.create_study(direction="minimize", sampler=sampler)
                    logging.info("Starting hyperparameter tuning")
                    study.optimize(self.objective, n_trials=100, show_progress_bar=True)

                    best_model, best_mae = self.train_and_log_model(model_name, study)

                    best_models.append(best_model)
                    best_maes.append(best_mae)

                self.log_ensemble_metrics(best_maes, best_models)

        except Exception as error_message:
            logging.error(f"Initiate model training failed with error: {error_message}")
            raise CustomException(error_message, sys) from error_message

    def feature_importance_plot(self, model, model_name):
        """
        Generate and save a feature importance plot.

        Args:
            model: Trained machine learning model.
            model_name (str): Name of the model.
        """
        try:
            logging.info("Executing feature importance plots")

            feature_importances = model.feature_importances_
            n_features = len(feature_importances)

            plt.figure(figsize=(10, 6))
            plt.barh(range(n_features), feature_importances, align="center")

            plt.yticks(np.arange(n_features), self.data.feature_names)

            plt.xlabel("Feature Importance")
            plt.ylabel("Feature")

            plt.title(f"{model_name} Model - Feature Importance")

            plt.tight_layout()

            os.makedirs(self.trainer_paths.explainability_path, exist_ok=True)
            plt.savefig(
                os.path.join(
                    self.trainer_paths.explainability_path,
                    f"{model_name}_feature_importance.png",
                )
            )
            plt.close()

        except Exception as error_message:
            logging.error(
                f"Feature importance plot generation failed with error: {error_message}"
            )
            raise CustomException(error_message, sys) from error_message

    def log_ensemble_metrics(self, best_maes, best_models):
        """
        Log ensemble metrics to MLflow.

        Args:
            best_maes (list): List of best mean absolute errors.
            best_models (list): List of best trained models.
        """
        try:
            logging.info("logging ensemble metrics")

            ensemble_predictions = np.zeros_like(self.data.y_test)
            total_weight = sum(1 / mae for mae in best_maes)
            weights = [1 / mae / total_weight for mae in best_maes]
            mlflow.log_params({"weights": weights})

            for model, weight in zip(best_models, weights):
                predictions = model.predict(self.data.x_test)
                ensemble_predictions += weight * predictions

            ensemble_mae = self.target_scaler.inverse_transform(
                mean_absolute_error(
                    self.data.y_test, ensemble_predictions, multioutput="raw_values"
                ).reshape(1, -1)
            ).flatten()

            mlflow.log_metric("ensemble_mae_oberflaechenspannung", ensemble_mae[0])
            mlflow.log_metric("ensemble_mae_anionischetenside", ensemble_mae[1])
            mlflow.log_metric("ensemble_mae_nichtionischentenside", ensemble_mae[2])
            mlflow.log_metric("ensemble_mae_total", np.mean(ensemble_mae))

        except Exception as error_message:
            logging.error(
                f"Logging ensemble metrics failed with error: {error_message}"
            )
            raise CustomException(error_message, sys) from error_message

    def train_and_log_model(self, model_name, study):
        """
        Train a model, log metrics and artifacts to MLflow.

        Args:
            model_name (str): Name of the model.
            study (optuna.Study): Optuna study object.

        Returns:
            tuple: Tuple containing the best trained model and its mean absolute error.
        """
        try:
            logging.info("Training best model")

            params_with_prefix = {
                f"{model_name}_{key}": value for key, value in study.best_params.items()
            }

            mlflow.log_params(params_with_prefix)
            mlflow.log_metric(f"{model_name}_val_total_mae", study.best_value)

            best_model = self.train_model(model_name, study.best_params)

            if model_name == "xgb":
                mlflow.xgboost.log_model(
                    xgb_model=best_model,
                    artifact_path=f"models/{model_name}",
                    registered_model_name=model_name,
                )

            elif model_name == "rf":
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path=f"models/{model_name}",
                    registered_model_name=model_name,
                )
            else:
                raise ValueError(f"Unsupported model name: {model_name}")

            preds = best_model.predict(self.data.x_test)

            mae = self.target_scaler.inverse_transform(
                mean_absolute_error(
                    self.data.y_test, preds, multioutput="raw_values"
                ).reshape(1, -1)
            ).flatten()

            mlflow.log_metric(f"{model_name}_mae_oberflaechenspannung", mae[0])
            mlflow.log_metric(f"{model_name}_mae_anionischetenside", mae[1])
            mlflow.log_metric(f"{model_name}_mae_nichtionischentenside", mae[2])
            mlflow.log_metric(f"{model_name}_mae_total", np.mean(mae))

            if self.data.feature_names is not None:
                self.feature_importance_plot(best_model, model_name)

                mlflow.log_artifacts(
                    self.trainer_paths.explainability_path,
                    artifact_path="explainability",
                )

            return (best_model, study.best_value)

        except Exception as error_message:
            logging.error(
                f"Training and logging best model failed with error: {error_message}"
            )
            raise CustomException(error_message, sys) from error_message
