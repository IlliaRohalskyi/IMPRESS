"""
Flask Application for Data Upload and Prediction

This module defines a Flask web application for uploading data, validating it
based on expected features, and generating predictions using a machine learning model ensemble.
"""
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from src.components.model_loader import get_models
from src.logger import logging
from src.pipelines.predict_pipeline import prediction_pipeline

app = Flask(__name__)

models_and_scalers = get_models()


@app.route("/")
def index():
    """
    Renders the index page for uploading data.

    Returns:
        Flask Response: The rendered HTML page.
    """
    return render_template("upload.html")


def validate_data(data, models_and_scalers_list):
    """
    Validates if the uploaded data format is correct based on the expected features.

    Args:
        data (pd.DataFrame): The uploaded data.
        models_and_scalers_list (List[ModelAndScalers]): List of models and scalers.

    Returns:
        bool: True if data format is correct, False otherwise.
    """
    features_required = []
    for model_and_scalers in models_and_scalers_list:
        cols = model_and_scalers.feature_scaler.feature_names_in_
        features = np.unique(np.array([string.split("_")[0] for string in cols]))
        features_required = np.union1d(features, features_required)

    return (
        np.all(np.isin(features_required, data.columns))
        and len(data["waschen"].unique()) == 1
        and "experimentnummer" not in data.columns
    )


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Handles the file upload and prediction generation.

    Returns:
        Flask Response: The prediction message or an error message.
    """
    uploaded_file = request.files["file"]

    if uploaded_file.filename != "":
        try:
            data = pd.read_csv(uploaded_file, sep=";")

            if validate_data(data, models_and_scalers):
                predictions = prediction_pipeline(models_and_scalers, data)
                message = (
                    f"Oberflächenspannung: {predictions[0][0]},\n"
                    f"anionische Tenside: {predictions[0][1]},\n"
                    f"nichtionische Tenside: {predictions[0][2]}"
                )
                return message
            return "DataFrame format is wrong"
        except Exception as error_message:  # pylint: disable=W0718
            logging.info(f"Error processing the file: {error_message}")
            return f"error: Error processing the file: {str(error_message)}"
    return "error: No file selected"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=443)
