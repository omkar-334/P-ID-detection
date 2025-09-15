import json
import math
import os

import cv2
import matplotlib.pyplot as plt
from huggingface_hub import (
    # create_repo,
    snapshot_download,
    upload_large_folder,
)


def upload():
    repo_id = "omkar334/PIDdataset"
    # create_repo(repo_id, repo_type="dataset", exist_ok=True)
    upload_large_folder(repo_id=repo_id, folder_path="compress", repo_type="dataset")


def download():
    # Hugging Face dataset repo ID
    repo_id = "omkar334/PIDdataset"  # replace with your repo
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir="dataset")


def plot_training_curves(trainer, save_path="training_curves.png"):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(trainer.history["train_loss"], label="Training Loss")
    if trainer.history["val_loss"]:
        plt.plot(trainer.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    # Note: mAP plotting would need to be implemented if metrics are tracked per epoch
    plt.text(
        0.5,
        0.5,
        "mAP tracking not implemented",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )
    plt.title("Mean Average Precision")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")

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


def sanitize(bbox, min_size=1.0):
    """
    Ensure x1 < x2, y1 < y2 and box has at least min_size width/height.
    """
    x1, y1, x2, y2 = bbox
    x1_, x2_ = min(x1, x2), max(x1, x2)
    y1_, y2_ = min(y1, y2), max(y1, y2)
    if x2_ - x1_ < min_size:
        x2_ = x1_ + min_size
    if y2_ - y1_ < min_size:
        y2_ = y1_ + min_size
    return [x1_, y1_, x2_, y2_]


ANN_DIR = "dataset/annotations"  # folder with {index}.json
THRESHOLD = 700  # diagonal cutoff


def classify_size(bbox, threshold=THRESHOLD):
    """
    Compute the diagonal length and classify as 'small' or 'large'.
    Returns both classification and diagonal length.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    diag = math.sqrt(w**2 + h**2)
    size = "small" if diag <= threshold else "large"
    return size, diag


def add_size_to_annotations():
    files = [f for f in os.listdir(ANN_DIR) if f.endswith(".json")]
    for fname in files:
        fpath = os.path.join(ANN_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)

        # Add size and diagonal fields to each object
        for obj in data.get("objects", []):
            if "bbox" in obj:
                size, diag = classify_size(obj["bbox"])
                obj["size"] = size
                obj["diag"] = diag

        # Overwrite JSON file with new fields
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Updated {fname}")


if __name__ == "__main__":
    add_size_to_annotations()
