import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    uploaded_file = request.files["file"]

    if uploaded_file.filename != "":
        try:
            df = pd.read_csv(uploaded_file)

            num_rows = len(df)

            return jsonify(
                {"message": f"File uploaded successfully. Number of rows: {num_rows}"}
            )
        except Exception as e:
            return jsonify({"error": f"Error processing the file: {str(e)}"})
    else:
        return jsonify({"error": "No file selected"})


if __name__ == "__main__":
    app.run()
