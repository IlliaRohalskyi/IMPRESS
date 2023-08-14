"""
Test Data Transformation.

This module contains tests for the DataTransformation class in the data_transformation module.
"""
import os
import random
import numpy as np
import pandas as pd
import pytest
from src.utils import get_project_root
from src.components.data_transformation import DataTransformation

@pytest.fixture
def data_transformation_obj():
    """
    Fixture for DataTransformation Object.

    This fixture creates and returns an instance of the DataTransformation class.

    Returns:
        DataTransformation: An instance of the DataTransformation class.
    """
    obj = DataTransformation()
    obj.transformation_config.feature_scaler_path = os.path.join(get_project_root(), 'tests',
                                                         'test_data', 'test_feature_scaler.pkl')
    obj.transformation_config.target_scaler_path = os.path.join(get_project_root(), 'tests',
                                                         'test_data', 'test_target_scaler.pkl')
    return obj

@pytest.mark.parametrize("_", range(100))
def test_data_transformation(_, data_transformation_obj):
    """
    Test Data Transformation Functionality.

    This test case verifies the functionality of the transform_data method
    in the DataTransformation class.

    Args:
        data_transformation_object (DataTransformation instance)
    Returns:
        None
    """
    num_online_samples = 1000
    num_offline_samples = 100

    random_offline_data = {
        'experimentnummer': [random.randint(1, 6) for _ in range(num_offline_samples)],
        'timestamp_probeentnahme': [random.randint(1e+5, 1e+9) for _ in range(num_offline_samples)],
        'timestamp_messung': [random.randint(1e+5, 1e+9) for _ in range(num_offline_samples)],
        'vorschlagnummer': [random.randint(-2, 7) for _ in range(num_offline_samples)],
        'bemerkungen': [random.choice(['S1', 'W1', 'Test']) for _ in range(num_offline_samples)],
        'aktivsauerstoff': [random.uniform(0.02, 0.12) for _ in range(num_offline_samples)],
        'anionischetenside': [random.uniform(0.26, 0.35) for _ in range(num_offline_samples)],
        'bsb': [random.randint(6, 15) for _ in range(num_offline_samples)],
        'carbonathaerte': [random.randint(35, 45) for _ in range(num_offline_samples)],
        'csb': [random.uniform(15, 45) for _ in range(num_offline_samples)],
        'leitfaehigkeit': [random.uniform(380, 550) for _ in range(num_offline_samples)],
        'nichtionischentenside': [random.uniform(0.02, 0.08) for _ in range(num_offline_samples)],
        'oberflaechenspannung': [random.randint(30, 39) for _ in range(num_offline_samples)],
        'peressigsaeure': [random.uniform(0.10, 0.15) for _ in range(num_offline_samples)],
        'ph': [random.uniform(6.7, 7.3) for _ in range(num_offline_samples)],
        'truebung': [random.uniform(8, 13) for _ in range(num_offline_samples)],
        'wasserstoffperoxid': [random.randint(35, 43) for _ in range(num_offline_samples)],
        'bsbeq': [random.randint(6, 15) for _ in range(num_offline_samples)],
        'reserv1': [random.uniform(0.02, 0.08) for _ in range(num_offline_samples)],
        'reserv2': [random.uniform(0.02, 0.06) for _ in range(num_offline_samples)],
    }

    random_online_data = {
        'experimentnummer': [random.randint(1, 6) for _ in range(num_online_samples)],
        'timestamp': [random.randint(1e+5, 1e+9) for _ in range(num_online_samples)],
        'waschen': [random.choice([0, 1]) for _ in range(num_online_samples)],
        'spuelen': [],
        'csbeq': [random.uniform(10, 55) for _ in range(num_online_samples)],
        'truebung': [random.uniform(5, 27) for _ in range(num_online_samples)],
        'druck1': [random.uniform(80, 120) for _ in range(num_online_samples)],
        'druck2': [random.uniform(160, 220) for _ in range(num_online_samples)],
        'druck3': [random.uniform(110, 170) for _ in range(num_online_samples)],
        'fluss1': [random.uniform(38, 55) for _ in range(num_online_samples)],
        'fluss2': [random.uniform(48, 65) for _ in range(num_online_samples)],
        'ph': [random.uniform(6.7, 7.5) for _ in range(num_online_samples)],
        'leitfaehigkeit': [random.uniform(380, 550) for _ in range(num_online_samples)],
        'alkalinitaet': [random.uniform(28, 45) for _ in range(num_online_samples)],
        'vorschlagsnummer': [random.randint(-2, 7) for _ in range(num_online_samples)],
        'bsbeq': [random.randint(6, 15) for _ in range(num_online_samples)],
        'abs254': [random.uniform(0.08, 0.35) for _ in range(num_online_samples)],
        'abs360': [random.uniform(0.05, 0.18) for _ in range(num_online_samples)],
        'abs210': [random.uniform(0.14, 0.28) for _ in range(num_online_samples)],
        'reserv1': [random.uniform(0.02, 0.12) for _ in range(num_online_samples)],
        'reserv2': [random.uniform(0.02, 0.08) for _ in range(num_online_samples)],
    }

    for waschen_value in random_online_data['waschen']:
        if waschen_value == 1:
            random_online_data['spuelen'].append(0)
        else:
            random_online_data['spuelen'].append(1)


    online_train = pd.DataFrame(random_online_data)
    offline_train = pd.DataFrame(random_offline_data)

    online_pred = pd.DataFrame(random_online_data)
    x_train, y_train, x_test, y_test = data_transformation_obj.initiate_data_transformation(online_train,
                                                                          offline_train)

    x_pred = data_transformation_obj.initiate_data_transformation(online_pred, None)

    # Test 1: Outputs are numpy arrays
    assert isinstance(x_train, np.ndarray), "x_train should be a numpy array"
    assert isinstance(y_train, np.ndarray), "y_train should be a numpy array"
    assert isinstance(x_test, np.ndarray), "x_test should be a numpy array"
    assert isinstance(y_test, np.ndarray), "y_test should be a numpy array"
    assert isinstance(x_pred, np.ndarray), "x_pred should be a numpy array"

    # Test 2: Float dtype
    assert x_train.dtype == np.float64, "x_train dtype should be float64"
    assert y_train.dtype == np.float64, "y_train dtype should be float64"
    assert x_test.dtype == np.float64, "x_test dtype should be float64"
    assert y_test.dtype == np.float64, "y_test dtype should be float64"
    assert x_pred.dtype == np.float64, "x_pred dtype should be float64"

    # Test 3: No NaN
    assert not np.any(np.isnan(x_train)), "x_train should not contain NaN values"
    assert not np.any(np.isnan(y_train)), "y_train should not contain NaN values"
    assert not np.any(np.isnan(x_test)), "x_test should not contain NaN values"
    assert not np.any(np.isnan(y_test)), "y_test should not contain NaN values"
    assert not np.any(np.isnan(x_pred)), "x_pred should not contain NaN values"

    # Test 4: Train data is between 0 and 1
    assert len(x_train[(x_train < 0) & (x_train > 1)]) == 0, "x_train values should be between 0 and 1"
    assert len(y_train[(y_train < 0) & (y_train > 1)]) == 0, "y_train values should be between 0 and 1"

    # Test 5: Check for duplicates in test and train
    assert len(x_train) == len(np.unique(x_train, axis=0)), "Duplicate rows in x_train"
    assert len(y_train) == len(np.unique(y_train, axis=0)), "Duplicate rows in y_train"
    assert len(x_test) == len(np.unique(x_test, axis=0)), "Duplicate rows in x_test"
    assert len(y_test) == len(np.unique(y_test, axis=0)), "Duplicate rows in y_test"

    # Test 6: Check train data points are not in test set and vice versa
    for row in x_test:
        assert not np.any(np.all(row == x_train, axis=1)), "x_train data points found in x_test"
    for row in y_test:
        assert not np.any(np.all(row == y_train, axis=1)), "y_train data points found in y_test"

    #Test 7: Check if pickle file is present, if yes, delete to check for next iteration
    feature_scaler_path = data_transformation_obj.transformation_config.feature_scaler_path
    target_scaler_path = data_transformation_obj.transformation_config.target_scaler_path
    
    assert os.path.exists(feature_scaler_path), f"Pickle file '{feature_scaler_path}' is not present"
    assert os.path.exists(target_scaler_path), f"Pickle file '{target_scaler_path}' is not present"

    os.remove(feature_scaler_path)
    os.remove(target_scaler_path)
