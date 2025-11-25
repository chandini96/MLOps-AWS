# MLOps Pipeline - AWS SageMaker

A production-ready, modular machine learning pipeline on AWS SageMaker with components for each stage of the ML lifecycle.

## 📦 Components

### 1. **components/data_fetcher/** - Data Acquisition
- Fetches data from CSV files or URLs
- Docker containerized
- S3 compatible

### 2. **components/preprocess/** - Data Preprocessing  
- Handles missing values, encoding, normalization
- Docker containerized
- Input/output via S3

### 3. **components/train/** - Model Training (SageMaker Compatible)
- RandomForest model training
- S3 data input/output
- SageMaker container compatible

### 4. **components/evaluate/** - Model Evaluation
- Multiple metrics (accuracy, precision, recall, F1, ROC AUC)
- Visualization (confusion matrix, ROC curves)
- Feature importance

### 5. **components/model_registry/** - Model Versioning
- Model registration and versioning
- Metadata tracking
- Model promotion to production

## 🏗️ Pipeline Architecture

```
┌──────────────┐
│ Data Fetch   │ → Fetches data from S3
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Preprocess   │ → Cleans and prepares data
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Train Model  │ → Trains RandomForest
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Evaluate     │ → Evaluates performance
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Register     │ → Registers in Model Registry
└──────────────┘
```

## 🚀 Quick Start

### Prerequisites

- AWS Account with appropriate permissions
- Docker installed
- Python 3.9+
- AWS CLI configured

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build Docker Images

```bash
# Build all components
make build

# Or build individually
make build-data        # Data fetcher
make build-preprocess  # Preprocessor
make build-train       # Trainer
make build-eval        # Evaluator
make build-registry    # Model registry
```

### 3. Test Components Locally

```bash
make test
```

### 4. Deploy to SageMaker

#### Option A: Using deploy script

```bash
export SAGEMAKER_ROLE="arn:aws:iam::ACCOUNT:role/SageMakerRole"
export S3_BUCKET="your-mlops-bucket"
export AWS_REGION="us-east-1"

# Build and push images, then deploy pipeline
bash deploy_pipeline.sh
```

#### Option B: Manual deployment

```bash
# 1. Update ECR_REGISTRY in pipeline.py with your ECR repository
export ECR_REGISTRY="YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com"

# 2. Deploy pipeline
python pipeline.py --deploy

# 3. Start execution
python pipeline.py --start
```

## 📂 Project Structure

```
mlops-aws/
├── components/
│   ├── data_fetcher/
│   │   ├── data_fetch.py
│   │   ├── Dockerfile
│   │   └── __init__.py
│   ├── preprocess/
│   │   ├── preprocess.py
│   │   ├── Dockerfile
│   │   └── __init__.py
│   ├── train/
│   │   ├── train.py
│   │   ├── Dockerfile
│   │   └── __init__.py
│   ├── evaluate/
│   │   ├── evaluate.py
│   │   ├── Dockerfile
│   │   └── __init__.py
│   └── model_registry/
│       ├── model_registry.py
│       ├── Dockerfile
│       └── __init__.py
├── pipeline.py              # SageMaker pipeline definition
├── deploy_pipeline.sh       # Automated deployment script
├── docker-compose.yml       # Local testing with docker-compose
├── Makefile                 # Build commands
├── requirements.txt         # Single requirements file for all components
├── .gitignore
└── README.md

# Generated directories (not in git)
├── data/                    # Local data storage
├── models/                  # Trained models
├── evaluation/              # Evaluation results
└── registry/                # Model registry
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
export SAGEMAKER_ROLE="arn:aws:iam::ACCOUNT:role/SageMakerRole"
export S3_BUCKET="your-mlops-bucket"
export AWS_REGION="us-east-1"

# ECR Configuration (after pushing images)
export ECR_REGISTRY="YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com"

# Optional
export PIPELINE_NAME="mlops-pipeline"
```

### Pipeline Parameters

Edit `pipeline.py` to customize:
- Processing instance types
- Training instance types  
- S3 paths for input/output
- ECR image URIs

## 🐳 Docker Usage

### Local Testing with Docker Compose

```bash
# Start all services
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Test Individual Components

```bash
# Data fetch
docker run -v $(pwd)/data:/app/data mlops-data-fetch

# Preprocess
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/processed_data:/app/processed_data mlops-preprocess

# Train (with AWS credentials)
docker run -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
           -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
           -e AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION \
           mlops-train --s3_input_path s3://bucket/data.csv \
                       --s3_output_path s3://bucket/model.joblib

# Evaluate
docker run -v $(pwd)/models:/app/models \
           -v $(pwd)/evaluation:/app/evaluation mlops-evaluate

# Model Registry
docker run -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
           -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
           -v $(pwd)/registry:/app/registry mlops-registry
```

## ☁️ AWS SageMaker Integration

### Pipeline Execution

The SageMaker pipeline automatically orchestrates all components:

1. **Data Fetch**: Downloads data from S3
2. **Preprocess**: Cleans and prepares training data
3. **Train**: Trains RandomForest on SageMaker
4. **Evaluate**: Computes evaluation metrics
5. **Register**: Registers model in SageMaker Model Registry

### Running in SageMaker

```python
import boto3

# Start pipeline execution
client = boto3.client('sagemaker', region_name='us-east-1')
client.start_pipeline_execution(
    PipelineName='MLOpsPipeline',
    PipelineParameters={
        'InputDataS3Uri': 's3://your-bucket/data/raw/dataset.csv',
        'ProcessingInstanceType': 'ml.m5.xlarge',
        'TrainingInstanceType': 'ml.m5.large',
        'ModelApprovalStatus': 'Approved'
    }
)
```

### View Pipeline in SageMaker Studio

1. Open SageMaker Studio
2. Navigate to **Pipelines** in the left sidebar
3. Select **MLOpsPipeline**
4. Click **Start execution**
5. Monitor execution status

### Using the CLI

```bash
# List pipeline executions
aws sagemaker list-pipeline-executions \
    --pipeline-name MLOpsPipeline \
    --region us-east-1

# Describe specific execution
aws sagemaker describe-pipeline-execution \
    --pipeline-execution-arn <execution-arn> \
    --region us-east-1
```

## 💻 Python Usage

### Run Components Individually

```python
from components.data_fetcher import DataFetcher
from components.preprocess import DataPreprocessor
from components.evaluate import ModelEvaluator
from components.model_registry import ModelRegistry

# 1. Fetch data
fetcher = DataFetcher(data_dir="data")
df = fetcher.fetch_from_csv("your_data.csv")

# 2. Preprocess
preprocessor = DataPreprocessor()
df_processed = preprocessor.preprocess(df)

# 3. Save processed data
fetcher.save_data(df_processed, "processed_data.csv")

# 4. Evaluate (example)
evaluator = ModelEvaluator()
# ... evaluate model

# 5. Register model
registry = ModelRegistry()
# ... register model
```

## 🧪 Testing

```bash
# Test all components
make test

# Test specific component
python -c "from components.data_fetcher import DataFetcher; print('✓ data_fetcher OK')"
python -c "from components.preprocess import DataPreprocessor; print('✓ preprocess OK')"
python -c "from components.evaluate import ModelEvaluator; print('✓ evaluate OK')"
python -c "from components.model_registry import ModelRegistry; print('✓ registry OK')"
```

## 📝 Features

- ✅ **Modular Design**: Independent, reusable components in separate folders
- ✅ **Containerized**: Dockerized for consistent environments
- ✅ **AWS Native**: Built specifically for SageMaker
- ✅ **Production-Ready**: Logging, error handling, configuration management
- ✅ **Single Requirements**: One `requirements.txt` for entire project
- ✅ **Orchestrated**: Complete SageMaker Pipeline definition
- ✅ **Version Control**: Model registry with versioning and metadata
- ✅ **Scalable**: Each component can be scaled independently

## 🔗 Useful Commands

```bash
# Make commands
make help              # Show all available commands
make build             # Build all Docker images
make up                # Start with docker-compose
make logs              # View logs
make down              # Stop containers
make clean             # Clean Docker images and containers
make test              # Test all components

# Pipeline commands
python pipeline.py --deploy      # Deploy pipeline to SageMaker
python pipeline.py --start       # Start pipeline execution
```

## 📊 Monitoring

### View Pipeline Status

```bash
aws sagemaker list-pipeline-executions \
    --pipeline-name MLOpsPipeline \
    --max-results 10
```

### CloudWatch Logs

Each step logs to CloudWatch Logs:
- `/aws/sagemaker/ProcessingJobs` - Processing steps
- `/aws/sagemaker/TrainingJobs` - Training jobs
- `/aws/sagemaker/Models` - Model registry

## 🔒 Security Best Practices

1. Use IAM roles with least privilege
2. Encrypt S3 buckets at rest
3. Use VPC endpoints for SageMaker
4. Rotate AWS credentials regularly
5. Enable CloudTrail for audit logging

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🔗 Useful Links

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Pipelines Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [Docker Documentation](https://docs.docker.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
