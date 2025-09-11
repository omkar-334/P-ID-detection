import time
from pathlib import Path

import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class Wrapper:
    """
    Flexible P&ID Detection Model Wrapper
    Supports multiple tasks and model architectures
    """

    def __init__(
        self,
        device: str = "cpu",
        task: str = "all",  # 'symbol', 'word', 'line', 'all'
        model_type: str = None,  # 'FasterRCNN', 'RetinaNet', 'MaskRCNN'
        models_dir: str = "models",
    ) -> None:
        self.device = device
        self.task = task
        self.model_type = model_type
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        # Configure task-specific classes
        if task == "all":
            self.num_classes = 4  # background + symbol + word + line
            self.class_names = ["background", "symbol", "word", "line"]
        else:
            self.num_classes = 2  # background + one category
            self.class_names = ["background", task]

        self.model = None
        self.config = None
        self.metrics = None

        # Build model
        self._create_model()

    def _create_model(self):
        """
        Create Faster R-CNN model with specified number of classes
        """
        print(f"Creating Faster R-CNN with {self.num_classes} classes")
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)  # pretrained backbone
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes
        )
        self.model.to(self.device)

    def save_model(self, name: str = None, config: dict = None, metrics: dict = None):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = name or f"{self.task}_{self.model_type}_{timestamp}"
        if not filename.endswith(".pth"):
            filename += ".pth"
        filepath = self.models_dir / filename

        save_data = {
            "model_state_dict": self.model.state_dict(),
            "task": self.task,
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "config": config or self.config,
            "metrics": metrics or self.metrics,
            "timestamp": timestamp,
        }

        torch.save(save_data, filepath)
        print(f"Model saved to: {filepath}")
        return str(filepath)

    def load_model(self, model_path: str):
        save_data = torch.load(model_path, map_location=self.device)
        self.task = save_data["task"]
        self.model_type = save_data["model_type"]
        self.num_classes = save_data["num_classes"]
        self.class_names = save_data["class_names"]
        self.config = save_data.get("config", None)
        self.metrics = save_data.get("metrics", None)

        # Re-create model architecture
        self._create_model()
        self.model.load_state_dict(save_data["model_state_dict"])
        self.model.to(self.device)
        print(f"Model loaded: {model_path}")
        print(
            f"Task: {self.task}, Type: {self.model_type}, Classes: {self.class_names}"
        )
