# PowerShell script to build and push Docker images to ECR
# Run this in PowerShell: .\build_and_push_components.ps1

$ErrorActionPreference = "Stop"

# ========= CONFIGURATION ==========
$AWS_REGION = "us-east-1"                     # change if needed
$REPO_NAME = "ml-pipeline-components"         # your unified ECR repo
$COMPONENTS = @("data_fetcher", "preprocess", "train", "evaluate", "model_registry")  # all pipeline components
# ==================================

# Step 1: Get your AWS account ID
$ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text).Trim()

Write-Host "AWS Account ID: $ACCOUNT_ID"
Write-Host "Region: $AWS_REGION"
Write-Host "ECR Repository: $REPO_NAME"
Write-Host "Components: $($COMPONENTS -join ', ')"
Write-Host "-------------------------------------------"

# Step 2: Authenticate Docker to ECR
Write-Host "Authenticating Docker to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Step 3: Loop through each component, build and push
foreach ($component in $COMPONENTS) {
    $IMAGE_URI = "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${component}"

    Write-Host "-------------------------------------------"
    Write-Host "Building image for component: $component"
    Write-Host "Image URI: $IMAGE_URI"
    Write-Host "-------------------------------------------"

    # Build from project root (so Docker can see requirements.txt and components/ paths)
    docker build -f ".\components\$component\Dockerfile" -t $component .
    docker tag "${component}:latest" $IMAGE_URI

    Write-Host "Pushing $IMAGE_URI ..."
    docker push $IMAGE_URI
    Write-Host "SUCCESS: $component pushed successfully!"
}

Write-Host "All components built and pushed successfully!"

