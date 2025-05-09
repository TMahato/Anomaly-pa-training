import os
import pandas as pd
import pickle
from decimal import Decimal, ROUND_HALF_UP
from pycaret.anomaly import setup, create_model, assign_model
import posixpath

def main():
    # Define constants (no env vars)
    guid = "4322"
    node_parameters = "Current,Speed,Temperature,Vibration"
    node_id = "id-1724928852427-264"
    algorithms = ["abod"]  # you can add more if needed

    print(f"Node Parameters: {node_parameters}")
    print(f"Node ID: {node_id}")
    print(f"Algorithms: {algorithms}")

    # Path to local dataset (mounted artifact)
    input_path = "/app/data/Anomalydata.csv"
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"Failed to read training data: {e}")
        raise

    # Training logic
    model_results = {}
    for algorithm in algorithms:
        try:
            print(f"🚀 Training model with algorithm: {algorithm}")
            s = setup(data=df, verbose=False)
            model = create_model(algorithm)
            result = assign_model(model)
            print("Model created and assigned")

            # Save model to /app/model (for AI Core to capture as output artifact)
            model_dir = '/app/model'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'{guid}_{algorithm}.pkl')
            print("model_path", model_path)

            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            print(f"Model saved to {model_path}")

            scores = result['Anomaly_Score']
            mean_normal = Decimal(float(scores[result['Anomaly'] == 0].mean())).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
            std_normal = Decimal(float(scores[result['Anomaly'] == 0].std())).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)

            model_results[algorithm] = {
                'mean_normal': mean_normal,
                'std_normal': std_normal,
                'model_path': model_path
            }

        except Exception as e:
            print(f"Error training model {algorithm}: {e}")
            raise

    # Save metadata
    metadata_path = os.path.join(model_dir, f'{guid}_metadata.pkl')
    print("metadata_path", metadata_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'guid': guid,
            'node_id': node_id,
            'node_parameters': node_parameters,
            'algorithms': algorithms,
            'model_results': model_results
        }, f)

    print(f"Training completed. Metadata saved at {metadata_path}")

if __name__ == "__main__":
    main()
