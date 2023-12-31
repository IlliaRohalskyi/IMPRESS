"""
Data Ingestion Module.

This module provides a class with a method for data ingestion tasks
"""
import os
import sys
from dataclasses import dataclass

import pandas as pd
import psycopg2

from src.exception import CustomException
from src.logger import logging
from src.utils import get_project_root


@dataclass
class DataIngestionConfig:
    """
    Configuration settings for data ingestion.

    This class represents configuration options for data ingestion tasks. It provides default values
    for the path of offline and online data files.

    Attributes:
        offline_data_path (str): The default path of the offline data file.
        online_data_path (str): The default path of the online data file.
    """

    offline_data_path: str = os.path.join(
        get_project_root(),
        "Dataset",
        "20220311_114449_impress_backup_offlinemessungen.csv",
    )
    online_data_path: str = os.path.join(
        get_project_root(), "Dataset", "20220311_114449_impress_backup_sensors.csv"
    )


class DataIngestion:
    """
    Data ingestion class.

    This class provides a method for data ingestion tasks.

    Attributes:
        ingestion_config (DataIngestionConfig): The configuration settings for data ingestion tasks.
    """

    def __init__(self):
        """
        Initialize the DataIngestion instance.

        Args:
            ingestion_config (DataIngestionConfig): Configuration settings for data ingestion tasks.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiate the data ingestion process.

        This method triggers the ingestion of data from online and offline sources as specified
        in the configuration. It returns the online and offline data.

        Returns:
            pandas.DataFrame: A DataFrame containing the ingested online data.
            pandas.DataFrame: A DataFrame containing the ingested offline data.
        """

        logging.info("Initiating data ingestion")
        try:
            logging.info("Reading data")

            offline_data = pd.read_csv(self.ingestion_config.offline_data_path, sep=";")

            online_data = pd.read_csv(self.ingestion_config.online_data_path, sep=";")

            logging.info("Data ingestion completed successfully")
            return online_data, offline_data

        except Exception as error_message:
            logging.error(f"Data ingestion failed with error: {error_message}")
            raise CustomException(error_message, sys) from error_message

    def get_sql_table(self, table_name):
        """
        Initiate the data ingestion process from a Postgres database.

        This method triggers the ingestion of data from the database

        Returns:
            pandas.DataFrame: A DataFrame containing the ingested data.
        """
        try:
            hostname = os.environ.get("DB_HOSTNAME")
            database_name = os.environ.get("DB_NAME")
            username = os.environ.get("DB_USERNAME")
            password = os.environ.get("DB_PASSWORD")

            connection = psycopg2.connect(
                host=hostname, database=database_name, user=username, password=password
            )

            query = f"SELECT * FROM {table_name};"

            return pd.read_sql_query(query, connection)

        except psycopg2.Error as error_message:
            logging.error(f"Error connecting to the database: {error_message}")
            raise CustomException(error_message, sys) from error_message

        finally:
            if connection:
                connection.close()
