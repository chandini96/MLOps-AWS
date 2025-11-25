import os
import tarfile
import logging
import json
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SageMaker directory structure
MODEL_DIR = "/opt/ml/processing/model"
INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"
EVAL_DIR = "/opt/ml/processing/evaluation"   # For SageMaker Metrics UI


def extract_model_artifact():
    logger.info("Searching for model.tar.gz...")

    tar_path = None
    for root, _, files in os.walk(MODEL_DIR):
        for f in files:
            if f.endswith(".tar.gz"):
                tar_path = os.path.join(root, f)
                break
        if tar_path:
            break

    if not tar_path:
        raise FileNotFoundError("No model.tar.gz found in model directory.")

    logger.info(f"Found model artifact: {tar_path}")
    logger.info("Extracting model.tar.gz...")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(MODEL_DIR)

    logger.info("Extraction complete.")


def locate_model_file():
    logger.info("Locating .joblib or .pkl model file...")

    candidates = []
    for root, _, files in os.walk(MODEL_DIR):
        for f in files:
            if f.endswith((".joblib", ".pkl")):
                candidates.append(os.path.join(root, f))

    if not candidates:
        raise FileNotFoundError("No model file (.pkl or .joblib) found after extraction.")

    if len(candidates) > 1:
        logger.warning(f"Multiple model files found, using the first one: {candidates}")

    model_path = candidates[0]
    logger.info(f"Using model file: {model_path}")
    return model_path


def load_test_data():
    logger.info("Loading test dataset...")

    test_file = None
    for f in os.listdir(INPUT_DIR):
        if f.lower().startswith("test") and f.lower().endswith(".csv"):
            test_file = os.path.join(INPUT_DIR, f)
            break

    if not test_file:
        raise FileNotFoundError("No test CSV found. Ensure the file name starts with 'test'.")

    logger.info(f"Found test dataset: {test_file}")
    return pd.read_csv(test_file)


def evaluate_model(model_path, test_df):
    logger.info("Evaluating model...")

    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1].astype(int)

    model = load(model_path)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Evaluation complete: accuracy={accuracy:.4f}")

    return {"accuracy": accuracy}


def save_results(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Standard JSON output
    output_path = os.path.join(OUTPUT_DIR, "evaluation.json")
    with open(output_path, "w") as f:
        json.dump(results, f)

    # SageMaker Studio metrics format
    sm_metrics = {
        "metrics": {
            "accuracy": {"value": results["accuracy"], "standard_deviation": 0.0}
        }
    }

    eval_path = os.path.join(EVAL_DIR, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(sm_metrics, f)

    logger.info(f"Saved evaluation results to: {output_path}")
    logger.info(f"Saved SageMaker metrics to: {eval_path}")


def main():
    logger.info("===== Starting SageMaker Evaluation Step =====")

    extract_model_artifact()
    model_path = locate_model_file()
    test_df = load_test_data()
    results = evaluate_model(model_path, test_df)
    save_results(results)

    logger.info("===== Evaluation Step Complete =====")


if __name__ == "__main__":
    main()
