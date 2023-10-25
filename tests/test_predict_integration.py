"""
Predict Pipeline Integration Tests

This module contains integration tests for the prediction pipeline components.
It tests the interaction between the data transformation and ensemble prediction tasks
to ensure that the pipeline generates predictions correctly and behaves as expected.
"""
import os
import shutil

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

from src.components.model_loader import get_models
from src.pipelines.predict_pipeline import prediction_pipeline
from src.utils import get_project_root


def test_pred_integration():
    """
    Runs the prediction pipeline on a test table.
    Checks whether archive table has records and deletes them.
    Checks whether pred table does not have any records and creates them.
    """
    pred_table_name = "test_online_data"
    archive_table_name = "test_archived_data"
    test_csv_path = os.path.join(
        get_project_root(), "tests", "test_data", "synthetic_online.csv"
    )

    get_models()
    prediction_pipeline(
        pred_table_name=pred_table_name, write_table_name=archive_table_name
    )

    hostname = os.environ.get("DB_HOSTNAME")
    database_name = os.environ.get("DB_NAME")
    username = os.environ.get("DB_USERNAME")
    password = os.environ.get("DB_PASSWORD")

    connection = psycopg2.connect(
        host=hostname, database=database_name, user=username, password=password
    )

    cursor = connection.cursor()

    cursor.execute(f"SELECT COUNT(*) FROM {pred_table_name}")
    assert cursor.fetchone()[0] == 0, "Pred table has entries"

    cursor.execute(f"SELECT COUNT(*) FROM {archive_table_name}")
    assert cursor.fetchone()[0] != 0, "Archive table does not have entries"

    delete_query = f"DELETE FROM {archive_table_name}"
    cursor.execute(delete_query)

    data = pd.read_csv(test_csv_path, delimiter=";")

    engine = create_engine(
        f"postgresql://{username}:{password}@{hostname}/{database_name}"
    )

    data.to_sql(pred_table_name, engine, if_exists="append", index=False)

    connection.commit()
    cursor.close()
    connection.close()

    shutil.rmtree(os.path.join(get_project_root(), "ml_downloads"))
