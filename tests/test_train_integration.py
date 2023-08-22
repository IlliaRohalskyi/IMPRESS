"""
Integration Test for End-to-End Training Pipeline

This script defines an integration test for the end-to-end training pipeline. It tests the data
ingestion, transformation, and model training components to ensure their compatibility and
expected behavior.

Usage: Run this script with pytest to perform the integration test on the training pipeline.
"""

import os
from unittest.mock import patch

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import (DataTransformation,
                                                TrainTestData)
from src.components.model_trainer import ModelTrainer
from src.utils import get_project_root


def test_end_to_end_pipeline():
    """
    Integration test function.

    This function tests the data ingestion, transformation, and model training components
    to ensure their compatibility and expected behavior.
    """
    data_ingestion_object = DataIngestion()

    offline_data_path = os.path.join(
        get_project_root(), "tests", "test_data", "synthetic_offline.csv"
    )
    online_data_path = os.path.join(
        get_project_root(), "tests", "test_data", "synthetic_online.csv"
    )

    data_ingestion_object.ingestion_config.offline_data_path = offline_data_path
    data_ingestion_object.ingestion_config.online_data_path = online_data_path

    online_data, offline_data = data_ingestion_object.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()

    feature_scaler_path = os.path.join(
        get_project_root(), "tests", "test_data", "test_feature_scaler.pkl"
    )
    target_scaler_path = os.path.join(
        get_project_root(), "tests", "test_data", "test_target_scaler.pkl"
    )

    data_transformation_obj.transformation_config.feature_scaler_path = (
        feature_scaler_path
    )
    data_transformation_obj.transformation_config.target_scaler_path = (
        target_scaler_path
    )

    train_test_obj = data_transformation_obj.initiate_data_transformation(
        online_data, offline_data
    )
    with patch("src.components.model_trainer.load_pickle"):
        model_trainer = ModelTrainer(train_test_obj)

    assert isinstance(online_data, pd.DataFrame)
    assert isinstance(offline_data, pd.DataFrame)
    assert isinstance(train_test_obj, TrainTestData)
    assert isinstance(model_trainer, ModelTrainer)

    assert os.path.exists(
        feature_scaler_path
    ), f"Pickle file '{feature_scaler_path}' do not exist"
    assert os.path.exists(
        target_scaler_path
    ), f"Pickle file '{target_scaler_path}' do not exist"

    os.remove(feature_scaler_path)
    os.remove(target_scaler_path)
