# datasets/texts.py
import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def preprocess_text():
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((32, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


class TextDataset(Dataset):
    def __init__(
        self,
        images_dir="dataset/images",
        ann_dir="dataset/annotations",
        transform=preprocess_text,
    ):
        self.images_dir = Path(images_dir)
        self.ann_dir = Path(ann_dir)
        self.transform = transform or preprocess_text()

        self.samples = []
        for ann_file in sorted(self.ann_dir.glob("*.json")):
            with open(ann_file) as f:
                ann = json.load(f)
            img_path = self.images_dir / f"{ann_file.stem}.jpg"
            for obj in ann["objects"]:
                if obj["category"] == "word":
                    self.samples.append({
                        "img_path": img_path,
                        "bbox": obj["bbox"],
                        "text": obj["text"],
                    })
        print(f"Loaded {len(self.samples)} word samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(str(s["img_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = map(int, s["bbox"])
        crop = img[y1:y2, x1:x2]

        # Optional: rotate vertical words
        h, w = crop.shape[:2]
        if h > 0 and w > 0 and (w / h) <= 0.5:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

        # Convert to tensor
        img = transforms.ToTensor()(crop)

        target = {
            "boxes": torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),  # 1 = "word"
            "image_id": torch.tensor([idx]),  # include image id
        }
        return img, target
