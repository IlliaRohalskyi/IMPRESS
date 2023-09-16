import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from src.components.model_loader import get_models
from src.pipelines.predict_pipeline import prediction_pipeline

app = Flask(__name__)

models_and_scalers = get_models()


@app.route("/")
def index():
    return render_template("upload.html")


def validate_data(df, models_and_scalers):
    features_required = []
    for model_and_scalers in models_and_scalers:
        cols = model_and_scalers.feature_scaler.feature_names_in_
        features = np.unique(np.array([string.split("_")[0] for string in cols]))
        features_required = np.union1d(features, features_required)

    return (
        np.all(np.isin(features_required, df.columns))
        and len(df["waschen"].unique()) == 1
        and "experimentnummer" not in df.columns
    )


@app.route("/upload", methods=["POST"])
def upload_file():
    uploaded_file = request.files["file"]

    if uploaded_file.filename != "":
        try:
            df = pd.read_csv(uploaded_file, sep=";")

            if validate_data(df, models_and_scalers):
                predictions = prediction_pipeline(models_and_scalers, df)
                message = "Oberflächenspannung: {0}, anionische Tenside: {1}, nichtionische Tenside: {2}".format(
                    predictions[0][0], predictions[0][1], predictions[0][2]
                )
                return message
            else:
                return "DataFrame format is wrong"
        except Exception as e:
            return f"error: Error processing the file: {str(e)}"
    else:
        return "error: No file selected"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=443)
