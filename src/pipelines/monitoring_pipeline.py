from prefect import flow, task
import os
from src.components.data_ingestion import DataIngestion
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently import ColumnMapping
from src.utils import get_project_root
from src.components.data_transformation import DataTransformation
from scipy.stats import ks_2samp
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import smtplib
import psycopg2
from src.exception import CustomException
from src.logger import logging
import sys

@task
def load_and_delete_data(table_name):
    ingestion_obj = DataIngestion()
    archived_data = ingestion_obj.get_sql_table(table_name=table_name)
    to_delete_count = int(len(archived_data) * 0.9)
    
    archived_data_selected = archived_data.iloc[:to_delete_count]
    train_data = ingestion_obj.initiate_data_ingestion()
    
    connection = psycopg2.connect(
        host=os.environ.get("DB_HOSTNAME"),
        database=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USERNAME"),
        password=os.environ.get("DB_PASSWORD")
    )

    cursor = connection.cursor()

    delete_query = f"DELETE FROM archived_data WHERE experimentnummer IN ({','.join(map(str, archived_data_selected['experimentnummer']))})"
    cursor.execute(delete_query)
    connection.commit()

    cursor.close()
    connection.close()
    
    return {'archived_data': archived_data_selected, 'train_data': train_data}


@task
def make_report(results):
    archived_data = results['archived_data']
    train_data = results['train_data']

    train_data_transformed = DataTransformation.initiate_data_transformation(train_data)
    archived_data.drop(columns=['experimentnummer'], inplace=True)
    
    targets = [
                "oberflaechenspannung",
                "anionischetenside",
                "nichtionischentenside",
            ]
    
    target_drifts = []
    feature_drifts = []

    for column in archived_data.columns():
        _, p_value = ks_2samp(archived_data[column], train_data_transformed[column])

        if p_value < 0.05:
            if column in targets:
                target_drifts.append(column)
            else: feature_drifts.append(column)
    output = {'target_drifts': target_drifts, 'feature_drifts': feature_drifts} if target_drifts or feature_drifts else {}
    if target_drifts:
        for target in target_drifts:
            column_mapping = ColumnMapping()
            column_mapping.target = target
            
            cur = train_data_transformed[[col for col in train_data_transformed.columns if col != target]]
            cur[target] = train_data_transformed[target]
            
            ref = archived_data[[col for col in archived_data.columns if col != target]]
            ref[target] = archived_data[target]

            report = Report(metrics=[DataQualityPreset(), DataDriftPreset(), TargetDriftPreset()])
            
            report.run(current_data=cur, reference_data=ref, column_mapping=column_mapping)
            
            dir_path = os.path.join(get_project_root(), 'reports')
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, f'{target}_report.html')
            report.save_html(file_path)

            output[f'{target}_report_path'] = file_path
            
    elif feature_drifts and not target_drifts:
        cur = train_data_transformed[[col for col in train_data_transformed.columns if col != target]]
        ref = archived_data[[col for col in archived_data.columns if col != target]]
        
        report = Report(metrics=[DataQualityPreset, DataDriftPreset()])
        report.run(current_data=cur, reference_data=ref)
        
        dir_path = os.path.join(get_project_root(), 'reports')
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f'drift_report.html')
        report.save_html(file_path)

        output[f'drift_report_path'] = file_path

        

@task
def alert(results, smtp_server, smtp_port):
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
        if key.endswith('_report_path'):
            with open(file_path, "rb") as report_file:
                drift_type = key.split('_')[0]

                attach = MIMEApplication(report_file.read(), _subtype="html")
                attach.add_header("Content-Disposition", "attachment", filename=f"{drift_type}_report.html")
                msg.attach(attach)
                
                text_message += f"{drift_type}, "

    text_message = text_message[:-2]
    msg.attach(MIMEText(text_message, 'plain'))

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, recipient_email, msg.as_string())
    server.quit()
    
@flow(name="monitoring_pipeline")
def monitoring_pipeline(smtp_server='smtp.office365.com', smtp_port=587, table_name='archived_data'):

    data_results = load_and_delete_data(table_name)
    report_results = make_report(data_results)
    if report_results:
        alert(report_results, smtp_server, smtp_port)
