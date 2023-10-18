"""
This module contains an integration test for the monitoring pipeline.
It tests the entire pipeline with a mocked SMTP server.
"""
import os
from unittest.mock import patch

import pandas as pd
from sqlalchemy import create_engine

from src.pipelines.monitoring_pipeline import monitoring_pipeline
from src.utils import get_project_root


@patch("src.pipelines.monitoring_pipeline.smtplib.SMTP")
def test_monitoring(_):
    """
    Integration test for the monitoring pipeline with a mocked SMTP server.

    This test verifies the successful execution of the monitoring pipeline using synthetic data.

    Args:
        _: Mocked SMTP server object (unused).
    """
    smtp_server = "localhost"
    smtp_port = 1025
    table_name = "test_archived_data"

    hostname = os.environ.get("DB_HOSTNAME")
    database_name = os.environ.get("DB_NAME")
    username = os.environ.get("DB_USERNAME")
    password = os.environ.get("DB_PASSWORD")

    data_path = os.path.join(
        get_project_root(), "tests", "test_data", "synthetic_archived_data.csv"
    )
    data = pd.read_csv(data_path, sep=";")
    engine = create_engine(
        f"postgresql://{username}:{password}@{hostname}/{database_name}"
    )

    data.to_sql(table_name, engine, if_exists="append", index=False)
    monitoring_pipeline(
        smtp_server=smtp_server, smtp_port=smtp_port, table_name=table_name
    )

    connection = engine.connect()
    delete_query = f"DELETE FROM {table_name}"
    connection.execute(delete_query)
    connection.close()
