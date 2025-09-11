import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class PIDJSONDataset(Dataset):
    """
    Dataset for P&ID diagrams.

    Modes:
      - "individual": each symbol class_id is a separate detection class
      - "unified": all symbols collapsed into one "symbol" class
    """

    def __init__(
        self,
        images_dir="images",
        ann_dir="annotations",
        transform=None,
        device="cpu",
        symbol_mode="individual",
    ):
        assert symbol_mode in ["individual", "unified"], (
            "symbol_mode must be 'individual' or 'unified'"
        )
        self.images_dir = Path(images_dir)
        self.ann_dir = Path(ann_dir)
        self.transform = transform
        self.device = device
        self.symbol_mode = symbol_mode

        # Collect files
        self.json_files = sorted(self.ann_dir.glob("*.json"))
        print(f"Found {len(self.json_files)} annotation files")
        self.ids = [f.stem for f in self.json_files]

        if symbol_mode == "individual":
            # Build symbol class mapping dynamically
            symbol_class_ids = set()
            for jf in self.json_files:
                with open(jf) as f:
                    ann = json.load(f)
                for obj in ann["objects"]:
                    if obj["category"] == "symbol":
                        symbol_class_ids.add(int(obj["class_id"]))
            self.symbol_class_ids = sorted(symbol_class_ids)
            self.symbol_to_label = {
                cid: i + 1 for i, cid in enumerate(self.symbol_class_ids)
            }
            self.word_label = len(self.symbol_class_ids) + 1
            self.line_label = len(self.symbol_class_ids) + 2
            self.num_classes = (
                len(self.symbol_class_ids) + 3
            )  # background + symbols + word + line
            self.class_names = (
                ["background"]
                + [f"symbol_{cid}" for cid in self.symbol_class_ids]
                + ["word", "line"]
            )
        else:  # unified
            self.symbol_to_label = None  # not needed
            self.symbol_label = 1
            self.word_label = 2
            self.line_label = 3
            self.num_classes = 4
            self.class_names = ["background", "symbol", "word", "line"]

        print(f"Loaded {len(self.ids)} diagrams")
        print(f"Mode: {self.symbol_mode}")
        print(f"Classes: {self.class_names} ({self.num_classes})")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def sanitize(bbox, min_size=1.0):
        """
        Ensure x1 < x2, y1 < y2 and box has at least min_size width/height
        """
        x1, y1, x2, y2 = bbox
        x1_, x2_ = min(x1, x2), max(x1, x2)
        y1_, y2_ = min(y1, y2), max(y1, y2)
        if x2_ - x1_ < min_size:
            x2_ = x1_ + min_size
        if y2_ - y1_ < min_size:
            y2_ = y1_ + min_size
        return [x1_, y1_, x2_, y2_]

    def __getitem__(self, idx):
        data_idx = self.ids[idx]

        # Load image
        img_path = self.images_dir / f"{data_idx}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotation
        ann_path = self.ann_dir / f"{data_idx}.json"
        with open(ann_path) as f:
            ann = json.load(f)

        boxes, labels = [], []
        for obj in ann["objects"]:
            if obj["category"] == "symbol":
                if self.symbol_mode == "individual":
                    cls_id = int(obj["class_id"])
                    if cls_id not in self.symbol_to_label:
                        continue
                    label = self.symbol_to_label[cls_id]
                else:  # unified
                    label = self.symbol_label
            elif obj["category"] == "word":
                label = self.word_label
            elif obj["category"] == "line":
                label = self.line_label
            else:
                continue
            boxes.append(self.sanitize(obj["bbox"]))
            labels.append(label)

        if len(boxes) == 0:
            return None

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)

        # Convert image to tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([int(data_idx)], device=self.device),
        }
        return image, target


def collate_fn(batch):
    """Custom collate function for detection dataloader"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return [], []
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_dataloaders(
    batch_size=4,
    train_split=0.7,
    val_split=0.15,
    num_workers=0,
    device="cpu",
    images_dir="images",
    ann_dir="annotations",
    symbol_mode="individual",
):
    dataset = PIDJSONDataset(
        images_dir=images_dir,
        ann_dir=ann_dir,
        device=device,
        symbol_mode=symbol_mode,
    )

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # --- Split ---
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    print(
        f"Splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    ds = PIDJSONDataset(
        images_dir="images",
        ann_dir="annotations",
        symbol_mode="unified",
    )
    print(f"Dataset size: {len(ds)}")
    print(f"Classes: {ds.class_names}")
    img, target = ds[0]
    print(f"Image shape: {img.shape}")
    print(f"Target: {target}")
