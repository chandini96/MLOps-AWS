#!/bin/bash
# Deploy SageMaker Pipeline
# This script handles the complete deployment process

set -e

echo "🚀 Starting SageMaker Pipeline Deployment..."

# Configuration
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
export SAGEMAKER_ROLE=${SAGEMAKER_ROLE:-"arn:aws:iam::ACCOUNT:role/SageMakerRole"}
export S3_BUCKET=${S3_BUCKET:-"your-mlops-bucket"}
export PIPELINE_NAME=${PIPELINE_NAME:-"mlops-pipeline"}

# Step 1: Build and push Docker images to ECR
echo "📦 Building and pushing Docker images to ECR..."

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"

# Create ECR repository if it doesn't exist
REPOS=("data-fetcher" "preprocess" "train" "evaluate" "model-registry")
for repo in "${REPOS[@]}"; do
    echo "Creating ECR repository: $repo"
    aws ecr describe-repositories --repository-names $repo || \
        aws ecr create-repository --repository-name $repo
    
    # Login to ECR
    aws ecr get-login-password --region ${AWS_DEFAULT_REGION} | \
        docker login --username AWS --password-stdin ${ECR_REGISTRY}
    
    # Build and push image
    echo "Building and pushing $repo..."
    docker build -f components/${repo//-/_}/Dockerfile -t $repo:latest .
    docker tag $repo:latest ${ECR_REGISTRY}/${repo}:latest
    docker push ${ECR_REGISTRY}/${repo}:latest
done

# Step 2: Update pipeline.py with ECR image URIs
echo "🔧 Updating pipeline with ECR image URIs..."
sed -i.bak "s|<your-ecr-repo>|${ECR_REGISTRY}|g" pipeline.py

# Step 3: Deploy pipeline
echo "📋 Deploying SageMaker pipeline..."
python pipeline.py

echo "✅ Deployment complete!"
echo "To start pipeline execution, run:"
echo "  python pipeline.py --start"

