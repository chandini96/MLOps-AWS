"""
Script to list and display registered models in SageMaker Model Registry
"""

import boto3
import json
import sys
from datetime import datetime
from typing import List, Dict, Any

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configuration
REGION = "us-east-1"
MODEL_PACKAGE_GROUP_NAME = "MLOpsModelPackageGroup"

# Initialize SageMaker client
sagemaker_client = boto3.client('sagemaker', region_name=REGION)


def list_model_package_groups():
    """List all model package groups"""
    print("\n" + "="*80)
    print("📦 MODEL PACKAGE GROUPS")
    print("="*80)
    
    try:
        response = sagemaker_client.list_model_package_groups()
        groups = response.get('ModelPackageGroupSummaryList', [])
        
        if not groups:
            print("No model package groups found.")
            return []
        
        for group in groups:
            print(f"\n📁 Group Name: {group['ModelPackageGroupName']}")
            print(f"   ARN: {group['ModelPackageGroupArn']}")
            print(f"   Created: {group.get('CreationTime', 'N/A')}")
            print(f"   Description: {group.get('ModelPackageGroupDescription', 'N/A')}")
        
        return groups
    except Exception as e:
        print(f"❌ Error listing model package groups: {str(e)}")
        return []


def list_registered_models(model_package_group_name: str = MODEL_PACKAGE_GROUP_NAME):
    """List all registered models in a model package group"""
    print("\n" + "="*80)
    print(f"📋 REGISTERED MODELS IN: {model_package_group_name}")
    print("="*80)
    
    try:
        # List model packages in the group
        paginator = sagemaker_client.get_paginator('list_model_packages')
        page_iterator = paginator.paginate(
            ModelPackageGroupName=model_package_group_name
        )
        
        models = []
        for page in page_iterator:
            models.extend(page.get('ModelPackageSummaryList', []))
        
        if not models:
            print(f"\n⚠️  No registered models found in '{model_package_group_name}'")
            print("   This could mean:")
            print("   - The pipeline hasn't been executed yet")
            print("   - The model registration step hasn't completed")
            print("   - The model package group name is different")
            return []
        
        print(f"\n✅ Found {len(models)} registered model(s):\n")
        
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x.get('CreationTime', datetime.min), reverse=True)
        
        for idx, model in enumerate(models, 1):
            print(f"{'─'*80}")
            print(f"Model #{idx}")
            print(f"{'─'*80}")
            # Handle both ModelPackageName and ModelPackageArn
            model_package_arn = model.get('ModelPackageArn') or model.get('ModelPackageName', 'N/A')
            model_package_name = model.get('ModelPackageName') or model_package_arn.split('/')[-1] if model_package_arn != 'N/A' else 'N/A'
            
            print(f"📦 Model Package Name: {model_package_name}")
            print(f"🆔 Model Package ARN: {model_package_arn}")
            print(f"📅 Created: {model.get('CreationTime', 'N/A')}")
            print(f"✅ Status: {model.get('ModelPackageStatus', 'N/A')}")
            print(f"📝 Description: {model.get('ModelPackageDescription', 'N/A')}")
            print(f"🏷️  Approval Status: {model.get('ModelApprovalStatus', 'N/A')}")
            
            # Get detailed information
            try:
                detail = sagemaker_client.describe_model_package(
                    ModelPackageName=model_package_arn
                )
                
                # Model metrics
                if 'ModelMetrics' in detail:
                    metrics = detail['ModelMetrics']
                    if 'ModelStatistics' in metrics:
                        stats = metrics['ModelStatistics']
                        s3_uri = stats.get('S3Uri', '')
                        print(f"📊 Model Metrics S3 URI: {s3_uri if s3_uri else 'N/A'}")
                        
                        # Try to fetch and display metrics if available
                        if s3_uri:
                            try:
                                import urllib.parse
                                from urllib.parse import urlparse
                                parsed = urlparse(s3_uri)
                                bucket = parsed.netloc
                                key = parsed.path.lstrip('/')
                                
                                # Remove filename if present, add evaluation.json
                                if not key.endswith('.json'):
                                    key = key.rstrip('/') + '/evaluation.json'
                                
                                s3_client = boto3.client('s3', region_name=REGION)
                                obj = s3_client.get_object(Bucket=bucket, Key=key)
                                metrics_data = json.loads(obj['Body'].read().decode('utf-8'))
                                
                                print(f"📈 Model Performance Metrics:")
                                if 'metrics' in metrics_data:
                                    for metric_name, metric_value in metrics_data['metrics'].items():
                                        if isinstance(metric_value, dict):
                                            value = metric_value.get('value', metric_value)
                                        else:
                                            value = metric_value
                                        print(f"   • {metric_name}: {value}")
                                elif 'accuracy' in metrics_data:
                                    print(f"   • accuracy: {metrics_data['accuracy']}")
                            except Exception as e:
                                # Silently fail if metrics can't be fetched
                                pass
                
                # Inference specification
                if 'InferenceSpecification' in detail:
                    inf_spec = detail['InferenceSpecification']
                    print(f"📥 Supported Content Types: {inf_spec.get('SupportedContentTypes', [])}")
                    print(f"📤 Supported Response Types: {inf_spec.get('SupportedResponseMIMETypes', [])}")
                    
                    # Supported instance types
                    if 'SupportedTransformInstanceTypes' in inf_spec:
                        print(f"🔄 Transform Instances: {inf_spec['SupportedTransformInstanceTypes']}")
                    if 'SupportedRealtimeInferenceInstanceTypes' in inf_spec:
                        print(f"⚡ Inference Instances: {inf_spec['SupportedRealtimeInferenceInstanceTypes']}")
                
                # Model artifacts
                if 'InferenceSpecification' in detail and 'Containers' in detail['InferenceSpecification']:
                    containers = detail['InferenceSpecification']['Containers']
                    if containers:
                        print(f"🐳 Container Image: {containers[0].get('Image', 'N/A')}")
                        if 'ModelDataUrl' in containers[0]:
                            print(f"📦 Model Data URL: {containers[0]['ModelDataUrl']}")
                
            except Exception as e:
                print(f"⚠️  Could not fetch detailed information: {str(e)}")
            
            print()
        
        return models
        
    except sagemaker_client.exceptions.ResourceNotFound:
        print(f"\n❌ Model package group '{model_package_group_name}' not found.")
        print("   Available groups:")
        list_model_package_groups()
        return []
    except Exception as e:
        print(f"❌ Error listing registered models: {str(e)}")
        return []


def get_model_details(model_package_arn: str):
    """Get detailed information about a specific model"""
    print("\n" + "="*80)
    print(f"🔍 MODEL DETAILS: {model_package_arn}")
    print("="*80)
    
    try:
        response = sagemaker_client.describe_model_package(
            ModelPackageName=model_package_arn
        )
        
        print("\n📋 Full Model Package Details:")
        print(json.dumps(response, indent=2, default=str))
        
        return response
    except Exception as e:
        print(f"❌ Error getting model details: {str(e)}")
        return None


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="List registered models in SageMaker Model Registry")
    parser.add_argument("--group", type=str, default=MODEL_PACKAGE_GROUP_NAME,
                        help="Model package group name (default: MLOpsModelPackageGroup)")
    parser.add_argument("--list-groups", action="store_true",
                        help="List all model package groups")
    parser.add_argument("--details", type=str,
                        help="Get detailed information for a specific model package ARN")
    
    args = parser.parse_args()
    
    if args.list_groups:
        list_model_package_groups()
    elif args.details:
        get_model_details(args.details)
    else:
        list_registered_models(args.group)
        
        # Also show available groups
        print("\n" + "="*80)
        print("💡 TIP: Use --list-groups to see all model package groups")
        print("💡 TIP: Use --details <model-package-arn> to see full details of a model")
        print("="*80)


if __name__ == "__main__":
    main()

