#!/bin/bash
set -e  # exit on error

# ========= CONFIGURATION ==========
AWS_REGION="us-east-1"                     # change if needed
REPO_NAME="ml-pipeline-components"         # your unified ECR repo
COMPONENTS=("data_fetcher" "preprocess" "train" "evaluate" "model_registry")  # all pipeline components
# ==================================

# Step 1: Get your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "AWS Account ID: $ACCOUNT_ID"
echo "Region: $AWS_REGION"
echo "ECR Repository: $REPO_NAME"
echo "Components: ${COMPONENTS[@]}"
echo "-------------------------------------------"

# Step 2: Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Step 3: Loop through each component, build and push
for component in "${COMPONENTS[@]}"; do
  IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${component}"

  echo "-------------------------------------------"
  echo " Building image for component: ${component}"
  echo " Image URI: ${IMAGE_URI}"
  echo "-------------------------------------------"

  # Build from project root (so Docker can see requirements.txt and components/ paths)
  docker build -f "./components/${component}/Dockerfile" -t ${component} .
  docker tag ${component}:latest ${IMAGE_URI}

  echo " Pushing ${IMAGE_URI} ..."
  docker push ${IMAGE_URI}
  echo " ${component} pushed successfully!"
done

echo " All components built and pushed successfully!"

