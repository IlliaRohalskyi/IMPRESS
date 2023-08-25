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

                washing_df = merged_data_final[merged_data_final["waschen"] == 1].drop(
                    column=["waschen"]
                )
                rinsing_df = merged_data_final[merged_data_final["waschen"] == 0].drop(
                    column=["waschen"]
                )

                waschen_train_data, waschen_test_data = train_test_split(
                    washing_df,
                    shuffle=True,
                    test_size=0.2,
                    random_state=42,
                )

                waschen_x_train = waschen_train_data.iloc[:, :-3]
                waschen_y_train = waschen_train_data.iloc[:, -3:]

                waschen_x_test = waschen_test_data.iloc[:, :-3]
                waschen_y_test = waschen_test_data.iloc[:, -3:]

                waschen_feature_scaler = MinMaxScaler()
                waschen_target_scaler = MinMaxScaler()

                waschen_x_train_scaled = waschen_feature_scaler.fit_transform(
                    waschen_x_train
                )
                waschen_y_train_scaled = waschen_target_scaler.fit_transform(
                    waschen_y_train
                )

                waschen_x_test_scaled = np.array(
                    waschen_feature_scaler.transform(waschen_x_test)
                )
                waschen_y_test_scaled = np.array(
                    waschen_target_scaler.transform(waschen_y_test)
                )

                os.makedirs(
                    os.path.dirname(
                        self.transformation_config.waschen_feature_scaler_path
                    ),
                    exist_ok=True,
                )

                os.makedirs(
                    os.path.dirname(
                        self.transformation_config.waschen_target_scaler_path
                    ),
                    exist_ok=True,
                )

                save_pickle(
                    waschen_feature_scaler,
                    self.transformation_config.waschen_feature_scaler_path,
                )
                save_pickle(
                    waschen_target_scaler,
                    self.transformation_config.waschen_target_scaler_path,
                )

                rinsing_train_data, rinsing_test_data = train_test_split(
                    rinsing_df,
                    shuffle=True,
                    test_size=0.2,
                    random_state=42,
                )

                rinsing_x_train = rinsing_train_data.iloc[:, :-3]
                rinsing_y_train = rinsing_train_data.iloc[:, -3:]

                rinsing_x_test = rinsing_test_data.iloc[:, :-3]
                rinsing_y_test = rinsing_test_data.iloc[:, -3:]

                rinsing_feature_scaler = MinMaxScaler()
                rinsing_target_scaler = MinMaxScaler()

                rinsing_x_train_scaled = rinsing_feature_scaler.fit_transform(
                    rinsing_x_train
                )
                rinsing_y_train_scaled = rinsing_target_scaler.fit_transform(
                    rinsing_y_train
                )

                rinsing_x_test_scaled = np.array(
                    rinsing_feature_scaler.transform(rinsing_x_test)
                )
                rinsing_y_test_scaled = np.array(
                    rinsing_target_scaler.transform(rinsing_y_test)
                )

                os.makedirs(
                    os.path.dirname(
                        self.transformation_config.rinsing_feature_scaler_path
                    ),
                    exist_ok=True,
                )

                os.makedirs(
                    os.path.dirname(
                        self.transformation_config.rinsing_target_scaler_path
                    ),
                    exist_ok=True,
                )

                save_pickle(
                    rinsing_feature_scaler,
                    self.transformation_config.rinsing_feature_scaler_path,
                )
                save_pickle(
                    rinsing_target_scaler,
                    self.transformation_config.rinsing_target_scaler_path,
                )

                return (
                    TrainTestData(
                        x_train=waschen_x_train_scaled,
                        y_train=waschen_y_train_scaled,
                        x_test=waschen_x_test_scaled,
                        y_test=waschen_y_test_scaled,
                        feature_names=merged_data_final.columns[:-3],
                    ),
                    TrainTestData(
                        x_train=rinsing_x_train_scaled,
                        y_train=rinsing_y_train_scaled,
                        x_test=rinsing_x_test_scaled,
                        y_test=rinsing_y_test_scaled,
                        feature_names=merged_data_final.columns[:-3],
                    ),
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
        logging.info("Processing online data")

        online_df = online_data[online_data["vorschlagsnummer"] >= 0]
        online_df.drop_duplicates(inplace=True)

        online_data_dropped = online_df.drop(
            columns=[
                "timestamp",
                "spuelen",
                "vorschlagsnummer",
                "reserv1",
                "reserv2",
                "druck2",
                "druck3",
                "fluss2",
            ]
        )

        stat_functions = [
            "mean",
            lambda x: 0 if len(x) == 1 else x.std(),
            "median",
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75),
            lambda x: (x.iloc[-1] - x.iloc[0]) / len(x),
            lambda x: x.autocorr(lag=50) if len(x) > 51 else 0,
            lambda x: x.diff().std() if len(x) > 1 else 0,
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
            else f"{col}_trend"
            if stat == "<lambda_3>"
            else f"{col}_autocorr"
            if stat == "<lambda_4>"
            else f"{col}_diff_std"
            if stat == "<lambda_5>"
            else f"{col}_{stat}"
            for col, stat in online_data_final.columns
        ]
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

        return offline_data_clean.drop(columns=["bemerkungen"])
