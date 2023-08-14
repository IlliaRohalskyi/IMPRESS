"""
Data Transformation Module.

This module provides a class with a method for data transformation tasks
"""
import sys
import os
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.logger import logging
from src.exception import CustomException
from src.utils import get_project_root, save_pickle, load_pickle

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
    scaler_path: str = os.path.join(get_project_root(), 'artifacts/data_processing/scaler.pkl')

class DataTransformation:
    """
    Data transformation class.
    
    This class provides a method for data transformation tasks.
    
    Attributes:
        transformation_config (DataTransformationConfig):
                    The configuration settings for data transformation tasks.
    
    Example:
        obj = DataTransformation()
        train_data, test_data = obj.initiate_data_transformation(online_data, offline_data, True)
    """
    def __init__(self):
        """
        Initialize the DataTransformation instance.

        Args:
            transformation_config (DataTransformationConfig): 
                        Configuration settings for data transformation tasks.
        """
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, online_data, offline_data, training_phase):
        """
        Initiate the data transformation process.

        This method triggers the transformation of data from online and offline ingested dataframes.
        It returns the train_data and test_data if training_phase = True. If training_phase = False,
        returns transformed data ready for prediction

        Returns:
            if training_phase = True:
                pandas.DataFrame: dataframe ready for training tasks
                pandas.DataFrame: dataframe ready for testing tasks
                
            if training_phase = False:
                pandas.DataFrame: dataframe ready for prediction tasks
        """
        logging.info("Initiating data transformation")

        try:
            def preprocess_online_data(online_data):
                logging.info("Processing online data")
                online_df = online_data[online_data['vorschlagsnummer'] >= 0]
                online_df.drop_duplicates(inplace=True)

                online_data_dropped = online_df.drop(columns=['timestamp', 'spuelen',
                                                            'vorschlagsnummer', 'reserv1',
                                                            'reserv2', 'druck2',
                                                            'druck3', 'fluss2'])

                stat_functions = ['mean', lambda x: 0 if len(x) == 1 else x.std(), 'median',
                                lambda x: x.quantile(0.25), lambda x: x.quantile(0.75),
                                lambda x: (x.iloc[-1] - x.iloc[0]) / len(x)]

                online_data_final = online_data_dropped.groupby(['experimentnummer',
                                                                'waschen']).agg(stat_functions)

                online_data_final.columns = [f'{col}_percentile25' if stat == '<lambda_0>' else
                                            f'{col}_percentile75' if stat == '<lambda_1>' else
                                            f'{col}_trend' if stat == '<lambda_2>' else
                                            f'{col}_{stat}' for col, stat
                                            in online_data_final.columns]
                return online_data_final

            def preprocess_offline_data(offline_data):
                logging.info("Processing offline data")

                offline_data.drop_duplicates(inplace=True)
                offline_grouped = offline_data.groupby(['experimentnummer',
                                                        'bemerkungen']).mean().reset_index()

                offline_grouped['waschen'] = offline_grouped['bemerkungen'].map({'S1': 0, 'W1': 1})

                offline_data_clean = offline_grouped[
                    offline_grouped['bemerkungen'].isin(['S1', 'W1'])
                    ]

                return offline_data_clean.drop(columns=['bemerkungen'])

            online_data_final = preprocess_online_data(online_data)
            offline_data_final = preprocess_offline_data(offline_data)

            logging.info("Merging offline and online data")

            offline_data_target = offline_data_final.loc[:, ['experimentnummer', 'waschen',
                                                             'oberflaechenspannung',
                                                             'anionischetenside',
                                                             'nichtionischentenside']]

            merged_data = pd.merge(online_data_final, offline_data_target,
                                   left_on=['experimentnummer', 'waschen'],
                                   right_on=['experimentnummer', 'waschen'],
                                   how='inner')

            merged_data_final = (
                merged_data.reset_index(drop=True).drop(columns=['experimentnummer'])
            )

            if training_phase:

                train_data, test_data = train_test_split(merged_data_final, shuffle=True,
                                                         stratify=merged_data_final.waschen,
                                                         test_size=0.1, random_state=42)

                scaler = MinMaxScaler()
                train_data_scaled = scaler.fit_transform(train_data)

                save_pickle(scaler, self.transformation_config.scaler_path)

                test_data_scaled = scaler.transform(test_data)

                return train_data_scaled, test_data_scaled

            scaler = load_pickle(self.transformation_config.scaler_path)
            return scaler.transform(merged_data_final)


        except Exception as error_message:
            logging.error(f"Data transformation failed with error: {error_message}")
            raise CustomException(error_message, sys) from error_message
