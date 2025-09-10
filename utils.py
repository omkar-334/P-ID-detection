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
