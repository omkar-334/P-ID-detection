import time
import warnings

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import PIDJSONDataset, get_dataloaders
from trainer import Trainer
from utils import evaluate_model, plot_training_curves

warnings.filterwarnings("ignore")

# Detect device
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
IMAGES_DIR = "dataset/images"
ANN_DIR = "dataset/annotations"
# ---------------- Configuration ----------------
config = {
    "batch_size": 4,
    "epochs": 10,
    "learning_rate": 0.001,
    "train_split": 0.7,
    "val_split": 0.2,
    "test_split": 0.1,
    "iou_threshold": 0.5,
    "confidence_threshold": 0.5,
}

# ---------------- Load Dataset ----------------
print("Loading dataset and creating dataloaders...")
train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=config["batch_size"],
    train_split=config["train_split"],
    val_split=config["val_split"],
    num_workers=0,
    device=device,
    images_dir=IMAGES_DIR,
    ann_dir=ANN_DIR,
)

# ---------------- Model ----------------

dataset = PIDJSONDataset(
    images_dir=IMAGES_DIR,
    ann_dir=ANN_DIR,
    device=device,
)
num_classes = len(dataset.class_names)  # background + word + line + symbols

print(f"Total classes (incl. background): {num_classes}")


# Load model with pretrained backbone only (no head)
model = fasterrcnn_resnet50_fpn(weights_backbone="DEFAULT", weights=None)

# Replace the classifier head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# ---------------- Trainer ----------------
trainer = Trainer(model, device=device, lr=config["learning_rate"])

# ---------------- Training ----------------
print("Starting training...")
trainer.train(train_loader, val_loader=val_loader, epochs=config["epochs"])

# ---------------- Evaluation ----------------
print("Evaluating on test set...")
metrics = evaluate_model(
    trainer,
    test_loader,
    iou_threshold=config["iou_threshold"],
    confidence_threshold=config["confidence_threshold"],
)

print("\n=== FINAL RESULTS ===")
print(f"mAP (mean Average Precision): {metrics['mAP']:.4f}")
print(f"Mean IoU: {metrics['mean_iou']:.4f}")
print(f"Mean Precision: {metrics['mean_precision']:.4f}")
print(f"Mean Recall: {metrics['mean_recall']:.4f}")
print(f"Mean F1-Score: {metrics['mean_f1']:.4f}")
print(f"Average Inference Time: {metrics['avg_inference_time']:.4f} sec/image")

# ---------------- Plot Training Curves ----------------
plot_training_curves(trainer, save_path="models/training_curves.png")

# ---------------- Save Model ----------------
now = time.strftime("%Y%m%d_%H%M%S")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": config,
        "metrics": metrics,
        "class_names": dataset.class_names,
    },
    f"models/pid_detection_model_{now}.pth",
)

print(f"\nModel saved as 'models/pid_detection_model_{now}.pth'")
print("Training curves saved as 'models/training_curves.png'")
