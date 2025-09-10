from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class Wrapper:
    """
    P&ID Detection Model Wrapper
    Handles model creation, loading, saving, and configuration
    Provides a clean interface without model.model confusion
    """

    def __init__(
        self, device: str = "cpu", num_classes: int = 4, models_dir: str = "models"
    ) -> None:
        self.device = device
        self.num_classes = num_classes
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.config = None
        self.class_names = None
        self.metrics = None

        # Create the model architecture
        self._create_model()

    def _create_model(self, weights_backbone: str = "DEFAULT") -> None:
        """
        Create a new Faster R-CNN model with the specified number of classes

        Args:
            weights_backbone: Backbone weights to use
        """
        # Load model with pretrained backbone only (no head)
        self.model = fasterrcnn_resnet50_fpn(
            weights_backbone=weights_backbone, weights=None
        )

        # Replace the classifier head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes
        )
        self.model.to(self.device)

    def save_model(
        self,
        config: dict[str, Any] | None = None,
        class_names: list | None = None,
        metrics: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> str:
        """
        Save model with all associated metadata

        Args:
            config: Training configuration
            class_names: List of class names
            metrics: Optional evaluation metrics
            name: Custom filename (without extension). If None, uses 'model_timestamp.pth'

        Returns:
            Path to saved model file
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = name or f"model_{timestamp}"
        filepath = self.models_dir / f"{filename}.pth"

        # Update instance variables if provided
        if config is not None:
            self.config = config
        if class_names is not None:
            self.class_names = class_names
        if metrics is not None:
            self.metrics = metrics

        save_data = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config or {},
            "class_names": self.class_names or [],
            "metrics": self.metrics,
            "timestamp": timestamp,
        }

        torch.save(save_data, filepath)
        print(f"Model saved to: {filepath}")
        return str(filepath)

    def load(self, model_path: str) -> dict[str, Any]:
        """
        Load a saved model with all metadata

        Args:
            model_path: Path to the saved model file

        Returns:
            Dictionary with config, class_names, metrics
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load saved data
        save_data = torch.load(model_path, map_location=self.device)

        # Extract components
        model_state_dict = save_data["model_state_dict"]
        config = save_data["config"]
        class_names = save_data["class_names"]
        metrics = save_data.get("metrics", None)

        # Update model with correct number of classes
        num_classes = len(class_names)
        self.num_classes = num_classes
        self._create_model()

        # Load state dict
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)

        # Store for future reference
        self.config = config
        self.class_names = class_names
        self.metrics = metrics

        print(f"Model loaded from: {model_path}")
        print(f"Classes: {class_names}")
        print(f"Number of classes: {num_classes}")

        return {"config": config, "class_names": class_names, "metrics": metrics}

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the currently loaded model"""
        return {
            "status": "Model loaded",
            "device": str(self.device),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "config": self.config,
            "metrics": self.metrics,
        }
