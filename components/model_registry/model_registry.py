"""
Model Registry Component
Manages model versioning, storage, and retrieval
"""

import os
import json
import logging
import boto3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """Class to manage model registry operations"""
    
    def __init__(self, registry_dir: str = "registry", s3_bucket: Optional[str] = None):
        """
        Initialize ModelRegistry
        
        Args:
            registry_dir: Local directory for model registry
            s3_bucket: Optional S3 bucket name for remote storage
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.s3_bucket = s3_bucket
        self.metadata_file = self.registry_dir / "model_metadata.json"
        
        # Initialize metadata if it doesn't exist
        if not self.metadata_file.exists():
            self._init_metadata()
        
        logger.info(f"Initialized ModelRegistry with directory: {self.registry_dir}")
    
    def _init_metadata(self):
        """Initialize metadata file"""
        metadata = {
            "models": {},
            "created_at": datetime.now().isoformat()
        }
        self._save_metadata(metadata)
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file"""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def _save_metadata(self, metadata: Dict):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def register_model(
        self,
        model_path: str,
        model_name: str,
        version: Optional[str] = None,
        description: str = "",
        tags: List[str] = None,
        metrics: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Register a model in the registry
        
        Args:
            model_path: Path to the model file
            model_name: Name of the model
            version: Model version (auto-generated if None)
            description: Model description
            tags: List of tags
            metrics: Training metrics
        
        Returns:
            Registration information
        """
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_id = f"{model_name}_v{version}"
        model_dir = self.registry_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file to registry
        registered_model_path = model_dir / "model.joblib"
        
        if os.path.exists(model_path):
            import shutil
            shutil.copy(model_path, registered_model_path)
            logger.info(f"Copied model from {model_path} to {registered_model_path}")
        
        # Create model metadata
        model_info = {
            "model_id": model_id,
            "model_name": model_name,
            "version": version,
            "registered_at": datetime.now().isoformat(),
            "model_path": str(registered_model_path),
            "description": description,
            "tags": tags or [],
            "metrics": metrics or {}
        }
        
        # Update registry metadata
        metadata = self._load_metadata()
        if model_name not in metadata["models"]:
            metadata["models"][model_name] = {}
        
        metadata["models"][model_name][version] = model_info
        self._save_metadata(metadata)
        
        logger.info(f"Registered model: {model_id}")
        
        # Optionally upload to S3
        if self.s3_bucket:
            self._upload_to_s3(model_id, registered_model_path)
        
        return model_info
    
    def get_model(self, model_name: str, version: Optional[str] = None):
        """
        Retrieve a model from the registry
        
        Args:
            model_name: Name of the model
            version: Model version (latest if None)
        
        Returns:
            Loaded model object
        """
        metadata = self._load_metadata()
        
        if model_name not in metadata["models"]:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        # Get latest version if not specified
        if version is None:
            versions = list(metadata["models"][model_name].keys())
            versions.sort(reverse=True)
            version = versions[0]
            logger.info(f"No version specified, using latest: {version}")
        
        if version not in metadata["models"][model_name]:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")
        
        model_info = metadata["models"][model_name][version]
        model_path = model_info["model_path"]
        
        # Download from S3 if needed
        if self.s3_bucket and not os.path.exists(model_path):
            self._download_from_s3(model_info["model_id"], model_path)
        
        # Load and return model
        model = joblib.load(model_path)
        logger.info(f"Loaded model: {model_info['model_id']}")
        
        return model, model_info
    
    def list_models(self) -> Dict[str, List[str]]:
        """
        List all models and their versions
        
        Returns:
            Dictionary mapping model names to versions
        """
        metadata = self._load_metadata()
        result = {}
        
        for model_name, versions in metadata["models"].items():
            result[model_name] = list(versions.keys())
        
        return result
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a specific model version
        
        Args:
            model_name: Name of the model
            version: Model version (latest if None)
        
        Returns:
            Model metadata
        """
        metadata = self._load_metadata()
        
        if model_name not in metadata["models"]:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        if version is None:
            versions = list(metadata["models"][model_name].keys())
            versions.sort(reverse=True)
            version = versions[0]
        
        if version not in metadata["models"][model_name]:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")
        
        return metadata["models"][model_name][version]
    
    def _upload_to_s3(self, model_id: str, model_path: Path):
        """Upload model to S3"""
        try:
            s3_client = boto3.client('s3')
            s3_key = f"models/{model_id}/model.joblib"
            s3_client.upload_file(str(model_path), self.s3_bucket, s3_key)
            logger.info(f"Uploaded {model_id} to s3://{self.s3_bucket}/{s3_key}")
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
    
    def _download_from_s3(self, model_id: str, model_path: Path):
        """Download model from S3"""
        try:
            s3_client = boto3.client('s3')
            s3_key = f"models/{model_id}/model.joblib"
            os.makedirs(model_path.parent, exist_ok=True)
            s3_client.download_file(self.s3_bucket, s3_key, str(model_path))
            logger.info(f"Downloaded {model_id} from S3")
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
    
    def promote_model(self, model_name: str, version: str, stage: str = "production"):
        """
        Promote a model to a specific stage
        
        Args:
            model_name: Name of the model
            version: Model version
            stage: Target stage (production, staging, etc.)
        """
        metadata = self._load_metadata()
        
        if model_name not in metadata["models"]:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        if version not in metadata["models"][model_name]:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")
        
        metadata["models"][model_name][version]["stage"] = stage
        metadata["models"][model_name][version]["promoted_at"] = datetime.now().isoformat()
        
        self._save_metadata(metadata)
        logger.info(f"Promoted {model_name} v{version} to {stage}")


def main():
    """Example usage of ModelRegistry"""
    registry = ModelRegistry()
    
    print("✔ Model registry component ready.")
    
    # Example usage:
    # registry.register_model(
    #     model_path="models/model.joblib",
    #     model_name="my_model",
    #     version="1.0",
    #     description="First production model",
    #     metrics={"accuracy": 0.95, "f1_score": 0.93}
    # )
    # 
    # model, info = registry.get_model("my_model", "1.0")
    # models = registry.list_models()


if __name__ == "__main__":
    main()

