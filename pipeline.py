"""
SageMaker MLOps Pipeline Orchestration
Integrates ModelTrainer, ModelEvaluator, and ModelRegistry into a complete pipeline
"""

import os
import boto3
import time
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.functions import Join
from sagemaker.session import Session


# ------------------ Configuration ------------------
SAGEMAKER_ROLE = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::269897083978:role/SageMakerExecRole-pro-ai")
S3_BUCKET = os.environ.get("S3_BUCKET", "mlops-data01")
REGION = os.environ.get("AWS_REGION", "us-east-1")

ECR_REGISTRY = os.environ.get("ECR_REGISTRY", "269897083978.dkr.ecr.us-east-1.amazonaws.com")
REPO_NAME = "ml-pipeline-components"

IMAGE_URI_DATA_FETCH = f"{ECR_REGISTRY}/{REPO_NAME}:data_fetcher"
IMAGE_URI_PREPROCESS = f"{ECR_REGISTRY}/{REPO_NAME}:preprocess"
IMAGE_URI_TRAIN = f"{ECR_REGISTRY}/{REPO_NAME}:train"
IMAGE_URI_EVALUATE = f"{ECR_REGISTRY}/{REPO_NAME}:evaluate"
IMAGE_URI_REGISTRY = f"{ECR_REGISTRY}/{REPO_NAME}:model_registry"

# Pipeline parameters
input_data_s3_uri = ParameterString(
    name="InputDataS3Uri", # This is the S3 URI of the input data
    default_value=f"s3://{S3_BUCKET}/Heart_Disease_Prediction.csv"
)
processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.large") # This is the instance type for the training job
model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")

# SageMaker session
sagemaker_session = Session(boto_session=boto3.Session(region_name=REGION))
print("Initializing SageMaker MLOps Pipeline...")

# ------------------ 1️⃣ Data Fetch Step ------------------
data_fetch_processor = ScriptProcessor(
    image_uri=IMAGE_URI_DATA_FETCH,
    role=SAGEMAKER_ROLE,
    instance_type=processing_instance_type,
    instance_count=1,
    command=["python"],
    sagemaker_session=sagemaker_session
)

step_data_fetch = ProcessingStep(
    name="DataFetch",
    processor=data_fetch_processor,
    code="components/data_fetcher/data_fetch.py",
    inputs=[
        ProcessingInput(input_name="raw_data",
                        source=input_data_s3_uri,
                        destination="/opt/ml/processing/input")
    ],
    outputs=[
        ProcessingOutput(output_name="fetched_data",
                         source="/opt/ml/processing/output",
                         destination=f"s3://{S3_BUCKET}/mlops-pipeline/data/fetched")
    ]
)

# ------------------ 2️⃣ Preprocessing Step ------------------
preprocess_processor = ScriptProcessor(
    image_uri=IMAGE_URI_PREPROCESS,
    role=SAGEMAKER_ROLE,
    instance_type=processing_instance_type,
    instance_count=1,
    command=["python"],
    sagemaker_session=sagemaker_session
)

step_preprocess = ProcessingStep(
    name="PreprocessData",
    processor=preprocess_processor,
    code="components/preprocess/preprocess.py",
    inputs=[
        ProcessingInput(
            input_name="raw_data",
            source=step_data_fetch.properties.ProcessingOutputConfig.Outputs["fetched_data"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train_data",
            source="/opt/ml/processing/output/train",
            destination=f"s3://{S3_BUCKET}/mlops-pipeline/data/train"
        ),
        ProcessingOutput(
            output_name="test_data",
            source="/opt/ml/processing/output/test",
            destination=f"s3://{S3_BUCKET}/mlops-pipeline/data/test"
        )
    ]
)

# ------------------ 3️⃣ Training Step ------------------
# Training script must use ModelTrainer and save model to /opt/ml/model/model.joblib
train_estimator = Estimator(
    image_uri=IMAGE_URI_TRAIN,
    role=SAGEMAKER_ROLE,
    instance_count=1,
    instance_type=training_instance_type,
    output_path=f"s3://{S3_BUCKET}/mlops-pipeline/models",
    sagemaker_session=sagemaker_session
)

step_train = TrainingStep(
    name="TrainModel",
    estimator=train_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri
        )
    }
)

# ------------------ 4️⃣ Evaluation Step ------------------
evaluate_processor = ScriptProcessor(
    image_uri=IMAGE_URI_EVALUATE,
    role=SAGEMAKER_ROLE,
    instance_type=processing_instance_type,
    instance_count=1,
    command=["python"],
    sagemaker_session=sagemaker_session
)

step_evaluate = ProcessingStep(
    name="EvaluateModel",
    processor=evaluate_processor,
    code="components/evaluate/evaluate.py",
    inputs=[
        ProcessingInput(
            input_name="model",
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            input_name="test_data",
            source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{S3_BUCKET}/mlops-pipeline/evaluation"
        ),
        ProcessingOutput(
            output_name="evaluation_output",
            source="/opt/ml/processing/output",
            destination=f"s3://{S3_BUCKET}/mlops-pipeline/evaluation/output"
        )
    ]
)

# ------------------ 5️⃣ Model Registration Step ------------------
# Assumes evaluate.py produces evaluation.json compatible with ModelMetrics
# The evaluation script saves to /opt/ml/processing/evaluation/evaluation.json
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(
            on="/",
            values=[
                step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                "evaluation.json"
            ]
        ),
        content_type="application/json"
    )
)

step_register = RegisterModel(
    name="RegisterModel",
    estimator=train_estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name="MLOpsModelPackageGroup",
    model_metrics=model_metrics,
    approval_status=model_approval_status
)

# ------------------ Define Pipeline ------------------
pipeline = Pipeline(
    name="MLOpsPipeline",
    steps=[step_data_fetch, step_preprocess, step_train, step_evaluate, step_register],
    parameters=[input_data_s3_uri, processing_instance_type, training_instance_type, model_approval_status]
)

print("Pipeline definition created successfully!")

# ------------------ Deployment & Execution Utilities ------------------
def deploy_pipeline():
    print(f"\nDeploying pipeline: {pipeline.name}")
    pipeline.upsert(role_arn=SAGEMAKER_ROLE)
    print("✅ Pipeline deployed successfully!")

def start_execution():
    print(f"\n🚀 Starting pipeline execution...")
    execution = pipeline.start()
    print(f"✅ Pipeline execution started!")
    print(f"📋 Execution ARN: {execution.arn}")
    print(f"🔗 View in console: https://console.aws.amazon.com/sagemaker/home?region={REGION}#/pipelines")
    return execution

def monitor_execution(execution_arn, poll_interval=30):
    """
    Monitor an existing pipeline execution by ARN
    """
    import datetime
    
    print(f"\n{'='*60}")
    print(f"📊 MONITORING PIPELINE EXECUTION")
    print(f"{'='*60}")
    print(f"Execution ARN: {execution_arn}")
    print(f"Polling every {poll_interval} seconds...")
    print(f"{'='*60}\n")
    
    # Use boto3 client for monitoring
    sagemaker_client = boto3.client('sagemaker', region_name=REGION)
    
    iteration = 0
    while True:
        iteration += 1
        try:
            # Describe pipeline execution using boto3
            desc = sagemaker_client.describe_pipeline_execution(
                PipelineExecutionArn=execution_arn
            )
            status = desc["PipelineExecutionStatus"]
            start_time = desc.get("PipelineExecutionStartTime", "")
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] Check #{iteration} - Pipeline Status: {status}")
            
            # List pipeline execution steps using boto3
            steps_response = sagemaker_client.list_pipeline_execution_steps(
                PipelineExecutionArn=execution_arn
            )
            steps = steps_response.get('PipelineExecutionSteps', [])
            if steps:
                print(f"\nStep Status:")
                for step in steps:
                    step_name = step.get('StepName', 'Unknown')
                    step_status = step.get('StepStatus', 'Unknown')
                    failure_reason = step.get('FailureReason', '')
                    
                    # Status indicator
                    if step_status == "Succeeded":
                        indicator = "✅"
                    elif step_status == "Failed":
                        indicator = "❌"
                    elif step_status == "InProgress" or step_status == "Executing":
                        indicator = "🔄"
                    else:
                        indicator = "⏳"
                    
                    print(f"  {indicator} {step_name}: {step_status}")
                    if failure_reason:
                        print(f"     Error: {failure_reason[:100]}...")
            else:
                print("  (No steps started yet)")
            
            if status in ["Succeeded", "Failed", "Stopped"]:
                print(f"\n{'='*60}")
                print(f"Pipeline execution finished with status: {status}")
                print(f"{'='*60}")
                if status == "Succeeded":
                    print("✅ Pipeline completed successfully!")
                elif status == "Failed":
                    print("❌ Pipeline execution failed. Check CloudWatch logs for details.")
                break
            
            print(f"\nWaiting {poll_interval} seconds before next check...")
            time.sleep(poll_interval)
            
        except Exception as e:
            print(f"⚠️  Error monitoring execution: {str(e)}")
            print(f"Continuing to monitor...")
            time.sleep(poll_interval)

# ------------------ Main ------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SageMaker MLOps Pipeline")
    parser.add_argument("--monitor", type=str, help="Monitor execution by ARN")
    parser.add_argument("--poll", type=int, default=30, help="Polling interval in seconds")
    args = parser.parse_args()

    if args.monitor:
        monitor_execution(args.monitor, poll_interval=args.poll)
    else:
        # Default behavior: Deploy and start pipeline execution
        # Deploy pipeline (creates/updates definition)
        deploy_pipeline()
        # Start execution
        exec_obj = start_execution()
        # Monitor execution
        monitor_execution(exec_obj.arn, poll_interval=args.poll)
   