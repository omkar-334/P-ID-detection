"""
RavenAI P&ID Annotation ML Model
A comprehensive solution for detecting and classifying annotated elements in P&ID diagrams

Dataset Structure:
- KeyValue: Project metadata and key-value pairs
- lines: Line elements with coordinates and properties
- lines2: Additional line data with numeric format
- linker: Connections between symbols and other elements
- symbols: Symbol annotations with bounding boxes and types
- Table: Table headers and structured data
- words: Text annotations with bounding boxes
"""

import warnings

import torch
import torch.optim as optim
from tqdm import tqdm

warnings.filterwarnings("ignore")


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


import torch


class Trainer:
    """
    Trainer for object detection models (Faster R-CNN, RetinaNet, Mask R-CNN)
    """

    def __init__(self, model, device="cpu", lr=0.001):
        self.model = model
        self.device = device
        self.lr = lr

        # Optimizer & LR scheduler
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
        )
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1
        )

        # History
        self.history = {"train_loss": [], "val_loss": []}

    def train(self, train_loader, val_loader=None, epochs=10):
        """
        Full training loop with optional validation
        """
        print(f"Training on {self.device} for {epochs} epochs")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss = self._train_one_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)

            self.lr_scheduler.step()

            if val_loss:
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f}")

    def _train_one_epoch(self, loader, epoch):
        """
        Train for one epoch
        """
        self.model.train()
        running_loss = 0.0
        batches = 0

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for images, targets in progress:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            self.optimizer.step()

            running_loss += losses.item()
            batches += 1

            progress.set_postfix({"loss": f"{losses.item():.4f}"})

        return running_loss / batches if batches > 0 else 0.0

    def validate(self, loader):
        """
        Validation pass — still computes loss (model must stay in train mode for this).
        """
        self.model.train()
        running_loss = 0.0
        batches = 0

        with torch.no_grad():
            for images, targets in loader:
                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                running_loss += losses.item()
                batches += 1

        return running_loss / batches if batches > 0 else 0.0

    def predict(self, images, confidence_threshold=0.5):
        """
        Run inference with confidence filtering
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            if not isinstance(images, list):
                images = [images]

            images = [img.to(self.device) for img in images]
            outputs = self.model(images)

            for output in outputs:
                keep = output["scores"] >= confidence_threshold
                pred = {
                    "boxes": output["boxes"][keep].cpu(),
                    "labels": output["labels"][keep].cpu(),
                    "scores": output["scores"][keep].cpu(),
                }
                predictions.append(pred)

        return predictions
