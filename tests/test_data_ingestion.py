"""
Test Data Ingestion.

This module contains test cases for the DataIngestion class from the data_ingestion module.
"""
import os
import pandas as pd
import pytest
from src.utils import get_project_root
from src.components.data_ingestion import DataIngestion

@pytest.fixture(name='data_ingestion_object')
def fixture_data_ingestion_object():
    """
    Fixture for DataIngestion Object.

    This fixture creates and returns an instance of the DataIngestion class.
    Changes the path of the offline and online data to the test files

    Returns:
        DataIngestion: An instance of the DataIngestion class.
    """
    data_ingestion_object = DataIngestion()
    offline_data_path = os.path.join(get_project_root(), 'tests', 'test_data',
                                     'synthetic_offline.csv')
    online_data_path = os.path.join(get_project_root(), 'tests', 'test_data',
                                    'synthetic_online.csv')
    data_ingestion_object.ingestion_config.offline_data_path = offline_data_path
    data_ingestion_object.ingestion_config.online_data_path = online_data_path
    return data_ingestion_object

def test_data_ingestion(data_ingestion_object):
    """
    Test Data Ingestion Functionality.

    This test case verifies the functionality of the initiate_data_ingestion method
    in the DataIngestion class.

    Args:
        data_ingestion_object (DataIngestion): An instance of the DataIngestion class.
    """
    online_data, offline_data = data_ingestion_object.initiate_data_ingestion()

    # Check if dataframes are not empty
    assert not online_data.empty, "Online data should not be empty"
    assert not offline_data.empty, "Offline data should not be empty"

    # Check data types
    assert isinstance(online_data, pd.DataFrame), "Online data should be a DataFrame"
    assert isinstance(offline_data, pd.DataFrame), "Offline data should be a DataFrame"
