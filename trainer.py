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

import time
import warnings

import numpy as np
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

    Args:
        wrapper: Wrapper instance containing the model
        device: Device to train on
        lr: Learning rate
    """

    def __init__(self, wrapper, device="cpu", lr=0.001):
        self.wrapper = wrapper
        self.model = wrapper.model  # Get the model from wrapper
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
        self.best_val_loss = float("inf")
        self.save_path = "models/best_model.pth"

    def save_checkpoint(self, name="best_model"):
        """Save current model as checkpoint using the wrapper"""
        # Update the wrapper's model with current state
        self.wrapper.model = self.model
        return self.wrapper.save_model(name=name)

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

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint()
                    print(f"Saved new best model with val loss {val_loss:.4f}")

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

    def evaluate(
        self, test_loader, iou_threshold=0.5, confidence_threshold=0.5, class_names=None
    ):
        """Evaluate the model and calculate metrics"""
        print("Evaluating model...")
        self.model.eval()

        all_predictions, all_targets, inference_times = [], [], []

        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Evaluating"):
                start_time = time.time()
                predictions = self.predict(images, confidence_threshold)
                inference_time = (time.time() - start_time) / len(images)
                inference_times.append(inference_time)

                all_predictions.extend(predictions)
                all_targets.extend(targets)

        metrics = self._calculate_metrics(
            all_predictions, all_targets, iou_threshold, class_names
        )
        metrics["avg_inference_time"] = np.mean(inference_times)
        return metrics

    def _calculate_metrics(
        self, predictions, targets, iou_threshold=0.5, class_names=None
    ):
        """Calculate detection metrics including mAP, precision, recall, F1"""
        if class_names is None:
            # Default class names if not provided
            class_names = ["background", "symbol", "word", "line"]

        metrics = {}
        num_classes = len(class_names)

        # Initialize counters
        tp_per_class = [0] * num_classes
        fp_per_class = [0] * num_classes
        fn_per_class = [0] * num_classes

        all_ious = []

        for pred, target in zip(predictions, targets):
            pred_boxes = (
                pred["boxes"].cpu().numpy() if len(pred["boxes"]) > 0 else np.array([])
            )
            pred_labels = (
                pred["labels"].cpu().numpy()
                if len(pred["labels"]) > 0
                else np.array([])
            )
            pred_scores = (
                pred["scores"].cpu().numpy()
                if len(pred["scores"]) > 0
                else np.array([])
            )

            target_boxes = target["boxes"].cpu().numpy()
            target_labels = target["labels"].cpu().numpy()

            # Match predictions with ground truth
            matched_gt = set()

            for i, (pred_box, pred_label, pred_score) in enumerate(
                zip(pred_boxes, pred_labels, pred_scores)
            ):
                # Ensure pred_label is within valid range
                if pred_label >= num_classes:
                    pred_label = num_classes - 1  # Clamp to last valid class

                best_iou = 0
                best_gt_idx = -1

                for j, (target_box, target_label) in enumerate(
                    zip(target_boxes, target_labels)
                ):
                    # Ensure target_label is within valid range
                    if target_label >= num_classes:
                        target_label = num_classes - 1  # Clamp to last valid class

                    if target_label == pred_label and j not in matched_gt:
                        iou = self._calculate_iou(pred_box, target_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j

                all_ious.append(best_iou)

                if best_iou >= iou_threshold and best_gt_idx != -1:
                    tp_per_class[pred_label] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp_per_class[pred_label] += 1

            # Count false negatives
            for j, target_label in enumerate(target_labels):
                if target_label >= num_classes:
                    target_label = num_classes - 1  # Clamp to last valid class
                if j not in matched_gt:
                    fn_per_class[target_label] += 1

        # Calculate metrics per class
        precisions = []
        recalls = []
        f1_scores = []

        for i in range(1, num_classes):  # Skip background class
            tp = tp_per_class[i]
            fp = fp_per_class[i]
            fn = fn_per_class[i]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

            print(
                f"{class_names[i]} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}"
            )

        # Overall metrics
        metrics["mAP"] = np.mean(precisions) if precisions else 0.0
        metrics["mean_precision"] = np.mean(precisions) if precisions else 0.0
        metrics["mean_recall"] = np.mean(recalls) if recalls else 0.0
        metrics["mean_f1"] = np.mean(f1_scores) if f1_scores else 0.0
        metrics["mean_iou"] = np.mean(all_ious) if all_ious else 0.0

        metrics["per_class_metrics"] = {
            "precision": precisions,
            "recall": recalls,
            "f1": f1_scores,
            "class_names": class_names[1:],  # Exclude background
        }

        return metrics

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes"""
        # Convert to [x1, y1, x2, y2] format if needed
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
