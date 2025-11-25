import argparse
import logging
import os
import joblib
import boto3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(s3_input_path: str, s3_output_path: str):
    """
    Train a simple RandomForest model on data from S3 and save the model back to S3.
    """
    # Local directories for training inside SageMaker container
    input_dir = "/opt/ml/input/data/train"
    model_dir = "/opt/ml/model"
    os.makedirs(model_dir, exist_ok=True)

    # Download data from S3
    logger.info(f"Downloading dataset from {s3_input_path} ...")
    s3 = boto3.client("s3")
    bucket, key = s3_input_path.replace("s3://", "").split("/", 1)
    local_path = os.path.join(input_dir, os.path.basename(key))
    os.makedirs(input_dir, exist_ok=True)
    s3.download_file(bucket, key, local_path)
    logger.info(f"Dataset downloaded to {local_path}")

    # Load dataset
    df = pd.read_csv(local_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train model
    logger.info("Training RandomForest model ...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    logger.info("Model training complete.")

    # Save model locally
    model_file = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_file)
    logger.info(f"Model saved locally at {model_file}")

    # Upload model back to S3
    bucket_out, key_out = s3_output_path.replace("s3://", "").split("/", 1)
    s3.upload_file(model_file, bucket_out, key_out)
    logger.info(f"Model uploaded to {s3_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--s3_input_path", type=str, required=True,
                        help="S3 path to input training data (CSV with 'target' column)")
    parser.add_argument("--s3_output_path", type=str, required=True,
                        help="S3 path to save trained model")

    args = parser.parse_args()
    train_model(args.s3_input_path, args.s3_output_path)
