import os
import pandas as pd
import pickle
from decimal import Decimal, ROUND_HALF_UP
from pycaret.anomaly import setup, create_model, assign_model
import shutil

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
        print(f"✅ Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Failed to read training data: {e}")
        raise

    for algorithm in algorithms:
        try:
            print(f"🚀 Training model with algorithm: {algorithm}")
            s = setup(data=df, verbose=False)
            model = create_model(algorithm)
            result = assign_model(model)
            print("✅ Model created and assigned")

            # Save to a writable directory first
            temp_model_dir = "/tmp/model"
            os.makedirs(temp_model_dir, exist_ok=True)
            temp_model_path = os.path.join(temp_model_dir, f"{guid}_{algorithm}.pkl")
            with open(temp_model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Model temporarily saved at {temp_model_path}")

            # Copy model to /app/model for artifact capture
            final_model_dir = "/app/model"
            os.makedirs(final_model_dir, exist_ok=True)
            final_model_path = os.path.join(final_model_dir, f"{guid}_{algorithm}.pkl")
            shutil.copy(temp_model_path, final_model_path)
            print(f"✅ Model copied to {final_model_path}")

            # Optional: show some stats
            scores = result['Anomaly_Score']
            mean_normal = Decimal(float(scores[result['Anomaly'] == 0].mean())).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
            std_normal = Decimal(float(scores[result['Anomaly'] == 0].std())).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)

            print(f"Mean score of normal: {mean_normal}")
            print(f"Std dev of normal: {std_normal}")

        except Exception as e:
            print(f"❌ Error training model {algorithm}: {e}")
            raise

    print("🎉 Training completed successfully.")

if __name__ == "__main__":
    main()
