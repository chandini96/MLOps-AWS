"""
Model Training Step — SageMaker-compatible
Trains a model using preprocessed data and saves it for SageMaker.
"""

import os
import sys
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # SageMaker Training Job input/output directories
    input_dir = "/opt/ml/input/data/train"  # Training data from preprocessing step
    output_dir = "/opt/ml/model"            # Where SageMaker expects the model
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Looking for training data in: {input_dir}")
    
    # Find the training CSV file (SageMaker may extract to subdirectories)
    train_file = None
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                train_file = os.path.join(root, file)
                logger.info(f"📁 Found training CSV: {train_file}")
                break
        if train_file:
            break
    
    if not train_file:
        logger.error(f" ERROR: No CSV file found in {input_dir}")
        logger.error(f"Contents of {input_dir}:")
        if os.path.exists(input_dir):
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                if os.path.isdir(item_path):
                    logger.error(f"  - {item}/ (directory)")
                else:
                    logger.error(f"  - {item}")
        sys.exit(1)
    
    # Load training data
    df = pd.read_csv(train_file)
    logger.info(f" Loaded training data: {df.shape}")
    
    # Identify target column
    if 'target' in df.columns:
        target_col = 'target'
    elif 'Target' in df.columns:
        target_col = 'Target'
    else:
        target_col = df.columns[-1]
        logger.info(f"Using last column '{target_col}' as target")
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info(f"Features: {X.shape}, Target: {y.shape}")
    
    # Convert target to integer for classification (handles float labels like 0.0, 1.0)
    y = y.astype(int)
    unique_values = y.nunique()
    logger.info(f"Target values: {sorted(y.unique())} ({unique_values} classes)")
    
    # Train Logistic Regression model for classification
    logger.info(" Starting model training...")
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    logger.info(" Model training completed")
    
    # Save model to SageMaker output path (required for SageMaker to upload to S3)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f" Model saved at {model_path}")
    
    # Verify model was saved
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path)
        logger.info(f"✅ Verified: Model file exists ({model_size} bytes)")
    else:
        logger.error(f"❌ ERROR: Model file was not created!")
        sys.exit(1)
    
    logger.info("✅ Training step completed successfully")

if __name__ == "__main__":
    main()
