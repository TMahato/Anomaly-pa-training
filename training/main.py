import os
import pandas as pd
import pickle
from io import BytesIO
from decimal import Decimal, ROUND_HALF_UP
import boto3
from pycaret.anomaly import setup, create_model, assign_model
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_s3_client():
    """Create an S3 client with credentials from environment variables"""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )

def main():
    # Get parameters from environment variables
    guid = os.getenv('GUID')
    plant = os.getenv('PLANT')
    node_parameters = os.getenv('NODE_PARAMETERS')
    node_id = os.getenv('NODE_ID')
    algorithms = os.getenv('ALGORITHM').split(',')
    
    # Create S3 client and get bucket name from .env
    s3_client = create_s3_client()
    bucket_name = os.getenv('BUCKET_NAME')
    
    # Define the training file path in S3
    training_file_key = f'anomaly/trainingData/{guid}'
    
    print(f"GUID: {guid}")
    print(f"Plant: {plant}")
    print(f"Node Parameters: {node_parameters}")
    print(f"Node ID: {node_id}")
    print(f"Algorithms: {algorithms}")
    print(f"Bucket Name: {bucket_name}")
    print(f"Training File Key: {training_file_key}")
    
    # Get training data from S3
    try:
        training_file = s3_client.get_object(Bucket=bucket_name, Key=training_file_key)
        csv_content = training_file['Body'].read().decode('utf-8')
        df = pd.read_csv(BytesIO(csv_content.encode('utf-8')))
        print(f"Successfully loaded training data: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading training data from S3: {e}")
        raise
    
    # Train models for each algorithm
    no_of_models = len(algorithms)
    individual_percentage = 100 / no_of_models
    trained_percentage = 0
    
    model_results = {}
    
    for algorithm in algorithms:
        try:
            print(f"Starting training for algorithm: {algorithm}")
            # Setup PyCaret
            s = setup(data=df, silent=True, session_id=123)
            
            # Create and train model
            model = create_model(algorithm)
            result = assign_model(model)
            
            # Save model locally
            model_dir = '/app/aicore-models'
            os.makedirs(model_dir, exist_ok=True)
            model_path = f'{model_dir}/{guid}_{algorithm}.pkl'
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"Model saved to {model_path}")
            
            # Calculate and log statistics
            scores = result['Anomaly_Score']
            mean_normal = Decimal(float(scores[result['Anomaly'] == 0].mean())).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
            std_normal = Decimal(float(scores[result['Anomaly'] == 0].std())).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
            
            print(f"Mean normal: {mean_normal}")
            print(f"Std normal: {std_normal}")
            
            # Store results
            model_results[algorithm] = {
                'mean_normal': mean_normal,
                'std_normal': std_normal,
                'model_path': model_path
            }
            
            # Update training progress
            trained_percentage += individual_percentage
            print(f"Training progress: {trained_percentage:.2f}%")
            
            print(f"Training completed for {algorithm}")
            
        except Exception as e:
            print(f"Error while training {algorithm}: {e}")
            raise
    
    # Save metadata about the models
    metadata_path = f'/app/aicore-models/{guid}_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'guid': guid,
            'plant': plant,
            'node_id': node_id,
            'node_parameters': node_parameters,
            'algorithms': algorithms,
            'model_results': model_results
        }, f)
    
    print(f"Model metadata saved to {metadata_path}")
    print("✅ All training completed successfully.")

if __name__ == "__main__":
    main()