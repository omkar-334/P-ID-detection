import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import (
    create_repo,
    snapshot_download,
    upload_large_folder,
)
from tqdm import tqdm


def upload():
    repo_id = "omkar334/PIDdataset"
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    upload_large_folder(repo_id=repo_id, folder_path="dataset", repo_type="dataset")


def download():
    # Hugging Face dataset repo ID
    repo_id = "omkar334/PIDdataset"  # replace with your repo
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir="dataset")


def calculate_iou(box1, box2):
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


def evaluate_model(model, test_loader, iou_threshold=0.5, confidence_threshold=0.5):
    """Evaluate the model and calculate metrics"""
    print("Evaluating model...")

    all_predictions = []
    all_targets = []
    inference_times = []

    model.model.eval()

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            start_time = time.time()

            predictions = model.predict(images, confidence_threshold)

            inference_time = (time.time() - start_time) / len(images)
            inference_times.append(inference_time)

            all_predictions.extend(predictions)
            all_targets.extend(targets)

    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets, iou_threshold)
    metrics["avg_inference_time"] = np.mean(inference_times)

    return metrics


def calculate_metrics(predictions, targets, iou_threshold=0.5):
    """Calculate detection metrics including mAP, precision, recall, F1"""
    metrics = {}

    # Per-class metrics
    class_names = ["background", "symbol", "word", "line", "table", "keyvalue"]
    num_classes = len(class_names)

    # Initialize counters
    tp_per_class = [0] * num_classes
    fp_per_class = [0] * num_classes
    fn_per_class = [0] * num_classes

    all_ious = []

    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"].numpy() if len(pred["boxes"]) > 0 else np.array([])
        pred_labels = (
            pred["labels"].numpy() if len(pred["labels"]) > 0 else np.array([])
        )
        pred_scores = (
            pred["scores"].numpy() if len(pred["scores"]) > 0 else np.array([])
        )

        target_boxes = target["boxes"].numpy()
        target_labels = target["labels"].numpy()

        # Match predictions with ground truth
        matched_gt = set()

        for i, (pred_box, pred_label, pred_score) in enumerate(
            zip(pred_boxes, pred_labels, pred_scores)
        ):
            best_iou = 0
            best_gt_idx = -1

            for j, (target_box, target_label) in enumerate(
                zip(target_boxes, target_labels)
            ):
                if target_label == pred_label and j not in matched_gt:
                    iou = calculate_iou(pred_box, target_box)
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


def plot_training_curves(model, save_path="training_curves.png"):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(model.train_losses, label="Training Loss")
    plt.plot(model.val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    if model.metrics_history:
        epochs = range(1, len(model.metrics_history) + 1)
        mAPs = [m["mAP"] for m in model.metrics_history]
        plt.plot(epochs, mAPs, "g-", label="mAP")
        plt.title("Mean Average Precision")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_bbox_on_image(bbox, image_path="boxes/0.jpg", zoom=False):
    """Plot bounding box on an image"""
    # Example bbox
    if not bbox:
        bbox = [1669, 767, 1702, 829]  # [x_min, y_min, x_max, y_max]
    x1, y1, x2, y2 = bbox

    image = cv2.imread(image_path)

    if zoom:
        margin = 50
        x1m, y1m = max(0, x1 - margin), max(0, y1 - margin)
        x2m, y2m = min(image.shape[1], x2 + margin), min(image.shape[0], y2 + margin)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop = image_rgb[y1m:y2m, x1m:x2m]

        plt.imshow(crop)
        plt.axis("off")
        plt.show()
    else:
        # Draw rectangle (BGR color, thickness=2)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Convert to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image_rgb)
        plt.axis("off")
        plt.show()
