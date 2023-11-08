"""
Data Transformation Module.

This module provides a class with a method for data transformation tasks
"""
import os
import sys
from dataclasses import dataclass
from typing import List

import miceforest as mf
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import get_project_root, save_pickle


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
                pd.DataFrame of features for prediction with experimentnummer and waschen values
        """
        logging.info("Initiating data transformation")

        try:
            online_data_final = self.preprocess_online_data(online_data)

            if offline_data is not None:
                offline_data_final = self.preprocess_offline_data(offline_data)

                logging.info("Merging offline and online data")

                merged_data_final = self.merge_data(
                    online_data_final, offline_data_final
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

            return online_data_final

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

            if "vorschlagsnummer" in online_data.columns:
                online_data = online_data[online_data["vorschlagsnummer"] >= 0]

            online_data = online_data.groupby(["experimentnummer", "waschen"]).head(700)

            online_data_dropped = online_data[
                [
                    "experimentnummer",
                    "waschen",
                    "truebung",
                    "druck1",
                    "fluss1",
                    "ph",
                    "leitfaehigkeit",
                    "timestamp",
                ]
            ]

            def apply_time_series_filter(data, window_length, polyorder):
                filtered_data = data.copy()
                for col in data.columns:
                    data_sorted = data.sort_values(by="timestamp", ascending=True)
                    filtered_data[col] = (
                        savgol_filter(data_sorted[col], window_length, polyorder)
                        if col != "timestamp"
                        else data_sorted["timestamp"]
                    )
                return filtered_data

            window_length = 200
            polyorder = 5

            filtered_df = (
                online_data_dropped.groupby(["experimentnummer", "waschen"])
                .apply(
                    lambda group: apply_time_series_filter(
                        group, window_length, polyorder
                    )
                )
                .reset_index()
            )

            filtered_df.drop(columns=["timestamp"], inplace=True)

            stat_functions = [
                "mean",
                lambda x: 0 if len(x) == 1 else x.std(),
                "median",
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.75),
            ]

            online_data_final = filtered_df.groupby(
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

        return online_data_final.reset_index()

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

            for col in [
                "oberflaechenspannung",
                "anionischetenside",
                "nichtionischentenside",
            ]:
                offline_data_clean[col] = np.where(
                    offline_data_clean[col] < 0, np.nan, offline_data_clean[col]
                )

            offline_data_clean.drop(columns=["bemerkungen"], inplace=True)

            kds = mf.ImputationKernel(
                offline_data_clean, save_all_iterations=True, random_state=42
            )

            kds.mice(10)

            return kds.complete_data()

        except Exception as error_message:
            logging.error(
                f"Offline data preprocessing failed with error: {error_message}"
            )
            raise CustomException(error_message, sys) from error_message

    def merge_data(self, preprocessed_online, preprocessed_offline):
        """
        Merge online and offline preprocessed data based on common columns.

        Args:
            preprocessed_online (pd.DataFrame): Preprocessed online data.
            preprocessed_offline (pd.DataFrame): Preprocessed offline data.

        Returns:
            pd.DataFrame: Merged data with common columns, excluding 'experimentnummer'.
        """
        offline_data_target = preprocessed_offline.loc[
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
            preprocessed_online,
            offline_data_target,
            left_on=["experimentnummer", "waschen"],
            right_on=["experimentnummer", "waschen"],
            how="inner",
        )

        merged_data_final = merged_data.reset_index(drop=True).drop(
            columns=["experimentnummer"]
        )
        return merged_data_final
