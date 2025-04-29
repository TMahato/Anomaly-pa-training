import os
import pandas as pd
from pycaret.anomaly import setup, create_model, assign_model
from decimal import Decimal, ROUND_HALF_UP
from sqlalchemy import text
from flask_sqlalchemy import SQLAlchemy
import joblib
from io import BytesIO
import boto3
import json
import base64
import requests

# Initialize SQLAlchemy
from flask import Flask
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_POOL_SIZE'] = 15
app.config['SQLALCHEMY_MAX_OVERFLOW'] = 50
app.config['SQLALCHEMY_POOL_TIMEOUT'] = 20
app.config['SQLALCHEMY_POOL_RECYCLE'] = 300
db = SQLAlchemy(app)

def create_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )

def sendPercentageMessage(node_Id, plant, anomaly_guid, trained_percentage):
    message = {
        "nodeId": node_Id,
        "plant": plant,
        "mainGuid": anomaly_guid,
        "percentage": trained_percentage
    }

    vcap_services = os.getenv('VCAP_SERVICES')
    if not vcap_services:
        raise Exception("VCAP_SERVICES environment variable not found")

    services = json.loads(vcap_services)
    em_credentials = services['enterprise-messaging'][0]['credentials']

    token_uri = em_credentials['management'][0]['oa2']['tokenendpoint'] + "?grant_type=client_credentials&response_type=token&"
    client_id = em_credentials['uaa']['clientid']
    client_secret = em_credentials['uaa']['clientsecret']

    message_queue_url = None
    for protocol in em_credentials['messaging']:
        if "httprest" in protocol['protocol']:
            message_queue_url = protocol['uri'] + os.getenv('message_queue')
            break

    if not message_queue_url:
        raise Exception("HTTP REST protocol not found in messaging protocols.")

    queue_name = "PredictiveAnalyticsTrainingPercentage"
    api_url = f"{message_queue_url}/{os.getenv('api_url1')}{queue_name}/{os.getenv('api_url2')}"

    token_headers = {'accept': 'application/json'}
    auth_string = f"{client_id}:{client_secret}"
    token_headers["Authorization"] = f"Basic {base64.b64encode(auth_string.encode()).decode()}"
    token_response = requests.post(token_uri, headers=token_headers)
    token_response.raise_for_status()
    access_token = token_response.json()["access_token"]

    headers = {
        'accept': 'application/json',
        'x-qos': '1',
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(message))
    response.raise_for_status()

def get_model_metadata(guid):
    query = text("SELECT PLANT, NODE_ID FROM PREDICTIVE_ANALYTICS_MODELS WHERE GUID = :guid")
    with db.session.begin():
        result = db.session.execute(query, {'guid': guid})
        row = result.fetchone()
    return (row[0], row[1]) if row else (None, None)

def update_model_status(guid, training_status, mean_normal, std_normal, model_name, model, s3_client, bucket_name):
    select_query = text("SELECT GUID FROM PREDICTIVE_ANALYTICS_ANOMALY_MODELS WHERE ANOMALY_GUID = :guid AND ALGORITHM = :model_name")
    with db.session.begin():
        result = db.session.execute(select_query, {'guid': guid, 'model_name': model_name})
        row = result.fetchone()
        if not row:
            raise ValueError(f"No GUID found for Model ({model_name}).")

        model_guid = row[0]
        update_query = text("""
            UPDATE PREDICTIVE_ANALYTICS_ANOMALY_MODELS
            SET TRAINING_STATUS = :training_status,
                ANOMALY_MEAN_NORMAL = :mean_normal,
                ANOMALY_STD_NORMAL = :std_normal
            WHERE GUID = :model_guid
        """)
        db.session.execute(update_query, {
            'training_status': training_status,
            'mean_normal': mean_normal,
            'std_normal': std_normal,
            'model_guid': model_guid
        })

        model_key = f"anomaly/model/{model_guid}"
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        s3_client.put_object(Bucket=bucket_name, Key=model_key, Body=buffer.getvalue())

def train():
    guid = os.getenv('GUID')
    plant = os.getenv('PLANT')
    node_Id = os.getenv('NODE_ID')
    algorithms = os.getenv('ALGORITHM').split(',')

    s3_client = create_s3_client()
    bucket_name = os.getenv('BUCKET_NAME')
    training_file_key = f'anomaly/trainingData/{guid}'

    training_file = s3_client.get_object(Bucket=bucket_name, Key=training_file_key)
    csv_content = training_file['Body'].read().decode('utf-8')
    df = pd.read_csv(BytesIO(csv_content.encode('utf-8')))

    no_of_models = len(algorithms)
    individual_percentage = 100 / no_of_models
    trained_percentage = 0

    for algorithm in algorithms:
        s = setup(data=df, silent=True, session_id=123)
        model = create_model(algorithm)
        result = assign_model(model)

        scores = result['Anomaly_Score']
        mean_normal = Decimal(float(scores[result['Anomaly'] == 0].mean())).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
        std_normal = Decimal(float(scores[result['Anomaly'] == 0].std())).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)

        update_model_status(guid, 4, mean_normal, std_normal, algorithm, model, s3_client, bucket_name)
        trained_percentage += individual_percentage
        sendPercentageMessage(node_Id, plant, guid, trained_percentage)

if __name__ == '__main__':
    train()