import time
from pathlib import Path

import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead

try:
    from clova.model import Model  # from ClovaAI repo
    from clova.utils import AttnLabelConverter

    clova = True
except ImportError:
    clova = False


class Wrapper:
    """
    Flexible P&ID Detection Model Wrapper
    Supports multiple tasks and model architectures
    """

    def __init__(
        self,
        device: str = "cpu",
        task: str = "all",  # 'symbol', 'word', 'line', 'all'
        model_type: str = "FasterRCNN",  # 'FasterRCNN', 'RetinaNet', 'OCR'
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

        self.create_model()

    def create_model(self):
        if self.model_type == "FasterRCNN":
            self.create_fasterrcnn()
        elif self.model_type == "RetinaNet":
            self.create_retinanet()
        elif self.model_type == "OCR":
            if clova:
                self.create_ocr()
            else:
                raise ValueError("ClovaAI not found")
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def create_fasterrcnn(self):
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

    def create_retinanet(self):
        """
        Create RetinaNet model with specified number of classes.

        Args:
            num_classes (int): number of classes including background
            device (str): "cuda" or "cpu"

        Returns:
            model (nn.Module): RetinaNet model
        """
        print(f"Creating RetinaNet with {self.num_classes} classes")

        # Load model with pretrained backbone
        self.model = retinanet_resnet50_fpn(weights="DEFAULT")

        # Replace classification head
        in_channels = self.model.head.classification_head.conv[0].in_channels
        num_anchors = self.model.head.classification_head.num_anchors
        self.model.head.classification_head = RetinaNetHead(
            in_channels, num_anchors, self.num_classes
        )

        self.model.to(self.device)

    def create_ocr(self):
        """
        Create OCR model with specified number of classes.
        """
        characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-./"
        converter = AttnLabelConverter(characters)
        num_class = len(converter.character)

        # --- Model ---
        self.model = Model(
            imgH=32,
            num_fiducial=20,
            input_channel=1,
            output_channel=512,
            hidden_size=256,
            num_class=num_class,
            batch_max_length=25,
            Transformation="TPS",
            FeatureExtraction="ResNet",
            SequenceModeling="BiLSTM",
            Prediction="Attn",
        ).to(self.device)

        print("Loading pretrained weights from models/TPS-ResNet-BiLSTM-Attn.pth")
        self.model.load_state_dict(
            torch.load("models/TPS-ResNet-BiLSTM-Attn.pth", map_location=self.device)
        )

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
        self.create_model()
        self.model.load_state_dict(save_data["model_state_dict"])
        self.model.to(self.device)
        print(f"Model loaded: {model_path}")
        print(
            f"Task: {self.task}, Type: {self.model_type}, Classes: {self.class_names}"
        )
