import pandas as pd
from pycaret.anomaly import setup, create_model, assign_model
import boto3
import os
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO
from dotenv import load_dotenv
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# S3 credentials from env
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
BUCKET_NAME = os.getenv('BUCKET_NAME')

# Get parameters from environment variables (set by Argo workflow)
PLANT = os.getenv('PLANT')
GUID = os.getenv('GUID')
NODE_PARAMETERS = os.getenv('NODE_PARAMETERS')
NODE_ID = os.getenv('NODE_ID')
ALGORITHM = os.getenv('ALGORITHM', 'iforest')  # Default to isolation forest if not specified

def train_model():
    logger.info(f"Starting anomaly detection training with algorithm: {ALGORITHM}")
    logger.info(f"Parameters: PLANT={PLANT}, GUID={GUID}, NODE_ID={NODE_ID}")
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        # Build S3 file key
        training_file_key = f'anomaly/trainingData/{GUID}'
        logger.info(f"Fetching training data from S3: {BUCKET_NAME}/{training_file_key}")
        
        # Fetch training data from S3
        training_file = s3_client.get_object(Bucket=BUCKET_NAME, Key=training_file_key)
        csv_content = training_file['Body'].read().decode('utf-8')
        df = pd.read_csv(BytesIO(csv_content.encode('utf-8')))
        
        logger.info(f"Training data loaded successfully with shape: {df.shape}")
        
        # Train model using PyCaret Anomaly Detection
        logger.info("Setting up PyCaret environment")
        setup(data=df, silent=True, session_id=123)
        
        logger.info(f"Creating model with algorithm: {ALGORITHM}")
        model = create_model(ALGORITHM)
        
        logger.info("Assigning anomaly scores")
        result = assign_model(model)
        
        # Calculate mean and std deviation of Anomaly Scores
        anomaly_scores = result['Anomaly_Score']
        mean_normal = Decimal(float(anomaly_scores[result['Anomaly'] == 0].mean())).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
        std_normal = Decimal(float(anomaly_scores[result['Anomaly'] == 0].std())).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
        
        logger.info(f"Model training completed. Mean normal: {mean_normal}, Std normal: {std_normal}")
        
        # Save trained model to model directory (will be picked up as artifact by Argo)
        from pycaret.anomaly import save_model
        os.makedirs('/app/model', exist_ok=True)
        model_path = f'/app/model/{ALGORITHM}_model'
        save_model(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata alongside the model
        metadata = {
            "message": "Model trained successfully",
            "plant": PLANT,
            "guid": GUID,
            "node_id": NODE_ID,
            "algorithm": ALGORITHM,
            "mean_normal": float(mean_normal),
            "std_normal": float(std_normal),
            "training_data_shape": df.shape
        }
        
        with open(f'/app/model/metadata.json', 'w') as f:
            json.dump(metadata, f)
        logger.info("Metadata saved alongside model")
        
        # Also save results to S3 for downstream processes
        results_bytes = json.dumps(metadata).encode('utf-8')
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=f'anomaly/results/{GUID}/results.json',
            Body=BytesIO(results_bytes)
        )
        logger.info(f"Results saved to S3: {BUCKET_NAME}/anomaly/results/{GUID}/results.json")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}", exc_info=True)
        # Write error to a file that can be picked up by the workflow
        with open('/app/model/error.txt', 'w') as f:
            f.write(f"Error: {str(e)}")
        return False

if __name__ == '__main__':
    logger.info("Script execution started")
    success = train_model()
    if success:
        logger.info("Script completed successfully")
        exit(0)
    else:
        logger.error("Script failed")
        exit(1)