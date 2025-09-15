# datasets/lines.py
import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from utils import sanitize


def preprocess_line(img):
    """
    Preprocessing for line sign/arrow detection:
      - Convert to tensor (C,H,W), normalize 0-1
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return F.to_tensor(img)


class LineSymbolDataset(Dataset):
    def __init__(
        self,
        images_dir="dataset/images",
        ann_dir="dataset/annotations",
        transforms=preprocess_line,
        categories=None,
    ):
        self.images_dir = Path(images_dir)
        self.ann_dir = Path(ann_dir)
        self.transforms = transforms
        self.categories = categories

        self.samples = []
        for ann_file in sorted(self.ann_dir.glob("*.json")):
            with open(ann_file) as f:
                ann = json.load(f)
            img_path = self.images_dir / f"{ann_file.stem}.jpg"
            boxes, labels = [], []
            for obj in ann["objects"]:
                if obj["category"] in self.categories:
                    boxes.append(sanitize(obj["bbox"]))
                    labels.append(self.categories[obj["category"]])
            if boxes:
                self.samples.append({
                    "img_path": str(img_path),
                    "boxes": boxes,
                    "labels": labels,
                })

        print(f"Loaded {len(self.samples)} line symbol samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["img_path"])
        img = self.transforms(img)

        boxes = torch.tensor(s["boxes"], dtype=torch.float32)
        labels = torch.tensor(s["labels"], dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return img, target
