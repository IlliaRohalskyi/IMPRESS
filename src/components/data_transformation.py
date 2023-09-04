"""
Data Transformation Module.

This module provides a class with a method for data transformation tasks
"""
import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import get_project_root, load_pickle, save_pickle


@dataclass
class TrainTestData:
    """
    Container for training and testing data along with feature names.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


@dataclass
class DataTransformationConfig:
    """
    Configuration settings for data transformation.

    This class represents configuration options for data transformation tasks.
    It provides default values for the path of scaler file.

    Attributes:
        scaler_path (str): The default path of the scaler file.

    Example:
        config = DataTransformationConfig()
        print(config.scaler_path)
    """

    feature_scaler_path: str = os.path.join(
        get_project_root(), "artifacts/data_processing/feature_scaler.pkl"
    )
    target_scaler_path: str = os.path.join(
        get_project_root(), "artifacts/data_processing/target_scaler.pkl"
    )


class DataTransformation:
    """
    Data transformation class.

    This class provides a method for data transformation tasks.

    Attributes:
        transformation_config (DataTransformationConfig):
                    The configuration settings for data transformation tasks.
    """

    def __init__(self):
        """
        Initialize the DataTransformation instance.

        Args:
            transformation_config (DataTransformationConfig):
                        Configuration settings for data transformation tasks.
        """
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(
        self, online_data: pd.DataFrame, offline_data: pd.DataFrame
    ):
        """
        Initiate the data transformation process.

        This method triggers the transformation of data from online and offline ingested dataframes.
        It returns the train_data and test_data if training_phase = True. If training_phase = False,
        returns transformed data ready for prediction

        Returns:
            if offline_data is not None:
                TrainTestData class instance with data ready for training

            else:
                np.array of features ready for prediction
        """
        logging.info("Initiating data transformation")

        try:
            online_data_final = self.preprocess_online_data(online_data)

            if offline_data is not None:
                offline_data_final = self.preprocess_offline_data(offline_data)

                logging.info("Merging offline and online data")

                offline_data_target = offline_data_final.loc[
                    :,
                    [
                        "experimentnummer",
                        "waschen",
                        "oberflaechenspannung",
                        "anionischetenside",
                        "nichtionischentenside",
                    ],
                ]

                merged_data = pd.merge(
                    online_data_final,
                    offline_data_target,
                    left_on=["experimentnummer", "waschen"],
                    right_on=["experimentnummer", "waschen"],
                    how="inner",
                )

                merged_data_final = merged_data.reset_index(drop=True).drop(
                    columns=["experimentnummer"]
                )

                train_data, test_data = train_test_split(
                    merged_data_final,
                    shuffle=True,
                    stratify=merged_data_final.waschen,
                    test_size=0.2,
                    random_state=42,
                )

                x_train = train_data.iloc[:, :-3]
                y_train = train_data.iloc[:, -3:]

                x_test = test_data.iloc[:, :-3]
                y_test = test_data.iloc[:, -3:]

                feature_scaler = MinMaxScaler()
                target_scaler = MinMaxScaler()

                x_train_scaled = feature_scaler.fit_transform(x_train)
                y_train_scaled = target_scaler.fit_transform(y_train)

                x_test_scaled = np.array(feature_scaler.transform(x_test))
                y_test_scaled = np.array(target_scaler.transform(y_test))

                os.makedirs(
                    os.path.dirname(self.transformation_config.feature_scaler_path),
                    exist_ok=True,
                )

                os.makedirs(
                    os.path.dirname(self.transformation_config.target_scaler_path),
                    exist_ok=True,
                )

                save_pickle(
                    feature_scaler, self.transformation_config.feature_scaler_path
                )
                save_pickle(
                    target_scaler, self.transformation_config.target_scaler_path
                )

                return TrainTestData(
                    x_train=x_train_scaled,
                    y_train=y_train_scaled,
                    x_test=x_test_scaled,
                    y_test=y_test_scaled,
                    feature_names=merged_data_final.columns[:-3],
                )

            feature_scaler = load_pickle(self.transformation_config.feature_scaler_path)
            return np.array(
                feature_scaler.transform(
                    online_data_final.drop(columns=["experimentnummer"])
                )
            )

        except Exception as error_message:
            logging.error(f"Data transformation failed with error: {error_message}")
            raise CustomException(error_message, sys) from error_message

    def preprocess_online_data(self, online_data):
        """
        Preprocess the online data.

        This method performs preprocessing steps on the given online data.

        Args:
            online_data (pd.DataFrame): The online data to be preprocessed.

        Returns:
            pd.DataFrame: The preprocessed online data.
        """
        try:
            logging.info("Processing online data")

            online_df = online_data[online_data["vorschlagsnummer"] >= 0]
            online_df.drop_duplicates(inplace=True)

            online_df.sort_values(
                by=["experimentnummer", "waschen", "timestamp"],
                ascending=[True, True, False],
                inplace=True,
            )

            data_points_extracted_df = online_df.groupby(
                ["experimentnummer", "waschen"]
            ).head(400)

            data_points_extracted_df.reset_index(drop=True, inplace=True)

            online_data_dropped = data_points_extracted_df.drop(
                columns=[
                    "timestamp",
                    "spuelen",
                    "vorschlagsnummer",
                    "reserv1",
                    "reserv2",
                    "druck2",
                    "druck3",
                    "fluss2",
                    "abs254",
                    "abs360",
                    "abs210",
                    "alkalinitaet",
                    "csbeq",
                    "bsbeq",
                ]
            )

            stat_functions = [
                "mean",
                lambda x: 0 if len(x) == 1 else x.std(),
                "median",
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.75),
            ]

            online_data_final = online_data_dropped.groupby(
                ["experimentnummer", "waschen"]
            ).agg(stat_functions)

            online_data_final.columns = [
                f"{col}_std"
                if stat == "<lambda_0>"
                else f"{col}_percentile25"
                if stat == "<lambda_1>"
                else f"{col}_percentile75"
                if stat == "<lambda_2>"
                else f"{col}_{stat}"
                for col, stat in online_data_final.columns
            ]
        except Exception as error_message:
            logging.error(
                f"Online data preprocessing failed with error: {error_message}"
            )
            raise CustomException(error_message, sys) from error_message

        return online_data_final

    def preprocess_offline_data(self, offline_data):
        """
        Preprocess the offline data.

        This method performs preprocessing steps on the given offline data.

        Args:
            offline_data (pd.DataFrame): The offline data to be preprocessed.

        Returns:
            pd.DataFrame: The preprocessed offline data.
        """
        try:
            logging.info("Processing offline data")

            offline_data.drop_duplicates(inplace=True)
            offline_grouped = (
                offline_data.groupby(["experimentnummer", "bemerkungen"])
                .mean()
                .reset_index()
            )

            offline_grouped["waschen"] = offline_grouped["bemerkungen"].map(
                {"S1": 0, "W1": 1}
            )

            offline_data_clean = offline_grouped[
                offline_grouped["bemerkungen"].isin(["S1", "W1"])
            ]

        except Exception as error_message:
            logging.error(
                f"Offline data preprocessing failed with error: {error_message}"
            )
            raise CustomException(error_message, sys) from error_message

        return offline_data_clean.drop(columns=["bemerkungen"])
