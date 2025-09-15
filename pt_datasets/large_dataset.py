import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from preprocess import preprocess_large_symbols
from torch.utils.data import Dataset

from utils import sanitize

ANN_DIR = "dataset/annotations"
IMG_DIR = "dataset/images"
SCALE_FACTOR = 1 / 8


def preprocess_large_symbols(img):
    """
    Preprocessing for symbols:
      - Grayscale
      - Adaptive threshold (binarization)
      - Convert to tensor
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    # optional dilation to preserve foreground before downscaling
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    # scale down
    new_size = (
        int(dilated.shape[1] * SCALE_FACTOR),
        int(dilated.shape[0] * SCALE_FACTOR),
    )
    resized = cv2.resize(dilated, new_size, interpolation=cv2.INTER_AREA)
    # adaptive threshold
    binary = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    tensor = torch.from_numpy(binary).unsqueeze(0).float() / 255.0
    return tensor


class LargeSymbolDataset(Dataset):
    def __init__(
        self, img_dir=IMG_DIR, ann_dir=ANN_DIR, transform=preprocess_large_symbols
    ):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform

        self.samples = []
        # load JSONs
        for fname in sorted(os.listdir(ann_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(ann_dir, fname)
            with open(fpath) as f:
                ann = json.load(f)

            # filter only large symbols
            has_large = any(
                obj.get("size") == "large" for obj in ann.get("objects", [])
            )
            if has_large:
                img_file = f"{ann['image_id']}.jpg"  # adjust if needed
                img_path = os.path.join(img_dir, img_file)
                self.samples.append({"img_path": img_path, "ann": ann})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        ann = sample["ann"]

        # Keep only large symbols
        boxes, labels = [], []
        for obj in ann.get("objects", []):
            if obj.get("category") != "symbol":
                continue
            if obj.get("size") != "large":
                continue
            bx1, by1, bx2, by2 = obj["bbox"]
            new_box = sanitize([
                bx1 * SCALE_FACTOR,
                by1 * SCALE_FACTOR,
                bx2 * SCALE_FACTOR,
                by2 * SCALE_FACTOR,
            ])
            boxes.append(new_box)
            labels.append(obj["class_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }
        img = self.transform(img)

        return img, target
