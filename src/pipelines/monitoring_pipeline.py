"""
This module defines a monitoring pipeline that loads, processes, and analyzes data
for drift detection and sends alert emails when drift is detected.

The pipeline consists of several tasks:

- `load_and_delete_data`: Loads data from a database table and deletes a portion of it.
- `make_report`: Generates drift reports for the loaded data, checking for data and target drift.
- `alert`: Sends alert emails with drift reports attached if drift is detected.
- `monitoring_pipeline`: Orchestrates the entire monitoring pipeline.

Usage:
- Ensure the necessary environment variables are set.
- Call the `monitoring_pipeline` function to run the entire monitoring pipeline.
"""

import os
import smtplib
import sys
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import psycopg2
from evidently import ColumnMapping
from evidently.metric_preset import (DataDriftPreset, DataQualityPreset,
                                     TargetDriftPreset)
from evidently.report import Report
from prefect import flow, task
from scipy.stats import ks_2samp

from src.components.data_ingestion import DataIngestion
from src.components.data_loader import load_dvc
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.utils import get_project_root


@task
def load_and_delete_data(table_name):
    """
    Loads data from a database table and deletes a portion of it.

    Args:
        table_name (str): The name of the SQL table to load and delete data from.

    Returns:
        dict: A dictionary containing the loaded data and retained online and offline data.
    """

    try:
        load_dvc()
        ingestion_obj = DataIngestion()
        archived_data = ingestion_obj.get_sql_table(table_name=table_name)
        if len(archived_data) < 10:
            return {}
        to_delete_count = int(len(archived_data) * 0.9)

        archived_data_selected = archived_data.iloc[:to_delete_count]
        online_data, offline_data = ingestion_obj.initiate_data_ingestion()

        connection = psycopg2.connect(
            host=os.environ.get("DB_HOSTNAME"),
            database=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USERNAME"),
            password=os.environ.get("DB_PASSWORD"),
        )

        cursor = connection.cursor()

        delete_query = (
            f"DELETE FROM {table_name} WHERE experimentnummer IN ("
            f"{','.join(map(str, archived_data_selected['experimentnummer']))})"
        )
        cursor.execute(delete_query)
        connection.commit()

        cursor.close()
        connection.close()

        return {
            "archived_data": archived_data_selected,
            "online_data": online_data,
            "offline_data": offline_data,
        }

    except CustomException as error_message:
        logging.error(f"Error during data load and delete: {error_message}")
        raise CustomException(error_message, sys) from error_message


@task
def make_report(results):
    """
    Generates drift reports for loaded data.

    Args:
        results (dict): A dictionary containing data for generating reports.

    Returns:
        dict: A dictionary with generated drift reports or an empty dictionary if no drift detected.
    """

    try:
        archived_data = results["archived_data"]
        online_data = results["online_data"]
        offline_data = results["offline_data"]
        transformation_obj = DataTransformation()

        online_preprocessed = transformation_obj.preprocess_online_data(online_data)
        offline_preprocessed = transformation_obj.preprocess_offline_data(offline_data)

        merged_data = transformation_obj.merge_data(
            online_preprocessed, offline_preprocessed
        )

        archived_data.drop(columns=["experimentnummer"], inplace=True)

        targets = [
            "oberflaechenspannung",
            "anionischetenside",
            "nichtionischentenside",
        ]

        target_drifts = []
        feature_drifts = []

        for column in archived_data.columns:
            _, p_value = ks_2samp(archived_data[column], merged_data[column])
            if p_value < 0.05:
                if column in targets:
                    target_drifts.append(column)
                else:
                    feature_drifts.append(column)
        output = (
            {"target_drifts": target_drifts, "feature_drifts": feature_drifts}
            if target_drifts or feature_drifts
            else {}
        )
        for target in target_drifts:
            if target_drifts:
                column_mapping = ColumnMapping()
                column_mapping.target = target

                cur = merged_data[[col for col in merged_data.columns if col != target]]
                cur[target] = merged_data[target]

                ref = archived_data[
                    [col for col in archived_data.columns if col != target]
                ]
                ref[target] = archived_data[target]

                report = Report(
                    metrics=[
                        DataQualityPreset(),
                        DataDriftPreset(),
                        TargetDriftPreset(),
                    ]
                )

                report.run(
                    current_data=cur, reference_data=ref, column_mapping=column_mapping
                )

                dir_path = os.path.join(get_project_root(), "reports")
                os.makedirs(dir_path, exist_ok=True)
                file_path = os.path.join(dir_path, f"{target}_report.html")
                report.save_html(file_path)

                output[f"{target}_report_path"] = file_path

            elif feature_drifts and not target_drifts:
                ref = merged_data[[col for col in merged_data.columns if col != target]]
                cur = archived_data[
                    [col for col in archived_data.columns if col != target]
                ]

                report = Report(metrics=[DataQualityPreset, DataDriftPreset()])
                report.run(current_data=cur, reference_data=ref)

                dir_path = os.path.join(get_project_root(), "reports")
                os.makedirs(dir_path, exist_ok=True)
                file_path = os.path.join(dir_path, "drift_report.html")
                report.save_html(file_path)

                output["drift_report_path"] = file_path
        return output
    except CustomException as error_message:
        logging.error(f"Error making reports: {error_message}")
        raise CustomException(error_message, sys) from error_message


@task
def alert(results, smtp_server, smtp_port):
    """
    Sends alert emails with drift reports attached.

    Args:
        results (dict): A dictionary containing report file paths.
        smtp_server (str): SMTP server address for sending emails.
        smtp_port (int): SMTP server port.
    """
    try:
        sender_email = os.environ.get("EMAIL_SENDER")
        sender_password = os.environ.get("EMAIL_PASS")
        recipient_email = os.environ.get("EMAIL_RECIPIENT")
        subject = "Evidently Drift Reports"

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject

        text_message = "Drift Type: "
        for key, file_path in results.items():
            if key.endswith("_report_path"):
                with open(file_path, "rb") as report_file:
                    drift_type = key.split("_")[0]

                    attach = MIMEApplication(report_file.read(), _subtype="html")
                    attach.add_header(
                        "Content-Disposition",
                        "attachment",
                        filename=f"{drift_type}_report.html",
                    )
                    msg.attach(attach)

                    text_message += f"{drift_type}, "

        text_message = text_message[:-2]
        msg.attach(MIMEText(text_message, "plain"))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
    except CustomException as error_message:
        logging.error(f"Error during alerting process: {error_message}")
        raise CustomException(error_message, sys) from error_message


@flow(name="monitoring_pipeline")
def monitoring_pipeline(
    smtp_server="smtp.office365.com", smtp_port=587, table_name="archived_data"
):
    """
    Orchestrates the monitoring pipeline.

    Args:
        smtp_server (str): SMTP server address for sending emails.
        smtp_port (int): SMTP server port.
        table_name (str): The name of the SQL table to load and delete data from.
    """
    data_results = load_and_delete_data(table_name)
    if data_results:
        report_results = make_report(data_results)
        if report_results:
            alert(report_results, smtp_server, smtp_port)


if __name__ == "__main__":
    monitoring_pipeline()
