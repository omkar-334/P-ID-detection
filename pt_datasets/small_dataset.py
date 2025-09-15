import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import sanitize


def preprocess_image(img):
    """
    Preprocessing for symbols:
      - Grayscale
      - Adaptive threshold (binarization)
      - Convert to tensor (1,H,W)
    """
    # Convert PIL to numpy if needed
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # To tensor (1,H,W), normalized [0,1]
    tensor = torch.from_numpy(binary).unsqueeze(0).float() / 255.0
    return tensor


class SmallSymbolDataset(Dataset):
    def __init__(
        self,
        img_dir="dataset/images",
        ann_dir="dataset/annotations",
        window_size=1024,
        stride=512,
        transform=preprocess_image,
    ):
        """
        Args:
            img_dir (str): Folder with images.
            ann_dir (str): Folder with annotation JSON files.
            window_size (int): Square patch size.
            stride (int): Stride for tiling patches.
            transform: Preprocessing function to apply to images.
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.window_size = window_size
        self.stride = stride
        self.transform = transform

        self.samples = []
        for ann_file in sorted(os.listdir(ann_dir)):
            if not ann_file.endswith(".json"):
                continue

            with open(os.path.join(ann_dir, ann_file)) as f:
                ann = json.load(f)

            img_file = f"{ann['image_id']}.jpg"  # adjust extension if needed
            img_path = os.path.join(img_dir, img_file)

            width, height = ann["width"], ann["height"]

            # Create patches for this image
            for x in range(0, width - window_size + 1, stride):
                for y in range(0, height - window_size + 1, stride):
                    patch_coords = (x, y, x + window_size, y + window_size)
                    self.samples.append({
                        "img_path": img_path,
                        "ann": ann,
                        "patch": patch_coords,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        x1, y1, x2, y2 = sample["patch"]

        # Crop patch
        img_patch = img.crop((x1, y1, x2, y2))

        # Extract boxes inside patch
        boxes, labels = [], []
        for obj in sample["ann"]["objects"]:
            if obj["category"] != "symbol":
                continue
            if obj.get("size") != "small":
                continue
            bx1, by1, bx2, by2 = obj["bbox"]

            # Keep any box that overlaps patch
            ix1 = max(bx1, x1)
            iy1 = max(by1, y1)
            ix2 = min(bx2, x2)
            iy2 = min(by2, y2)

            if ix1 < ix2 and iy1 < iy2:
                new_box = [ix1 - x1, iy1 - y1, ix2 - x1, iy2 - y1]
                new_box = sanitize(new_box)
                boxes.append(new_box)
                labels.append(obj["class_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }
        img_patch = self.transform(img_patch)

        return img_patch, target


def visualize_patch(img_tensor: torch.Tensor, target: dict, save_path=None):
    """
    Visualize a patch with bounding boxes.

    Args:
        img_tensor (torch.Tensor): Shape (1,H,W) or (3,H,W), values in [0,1].
        target (dict): Contains 'boxes' and 'labels'.
        save_path (str, optional): If given, saves the image instead of showing.
    """
    # Convert to numpy (0-255 uint8)
    img_np = img_tensor.mul(255).clamp(0, 255).byte().cpu().numpy()

    if img_np.ndim == 3:  # (C,H,W)
        img_np = np.transpose(img_np, (1, 2, 0))  # -> (H,W,C)
    else:  # (1,H,W) -> (H,W)
        img_np = img_np.squeeze(0)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # Enforce 3-channel BGR uint8 C-contiguous
    if img_np.ndim == 2 or img_np.shape[2] == 1:  # safety
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    img_np = np.ascontiguousarray(img_np, dtype=np.uint8)

    # Draw boxes
    for box in target["boxes"]:
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show or save
    if save_path:
        cv2.imwrite(save_path, img_np)
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = SmallSymbolDataset(
        img_dir="dataset/images",
        ann_dir="dataset/annotations",
        window_size=1024,
        stride=512,
        transform=preprocess_image,
    )

    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )

    # Quick check
    for imgs, targets in dataloader:
        for img_patch, target in zip(imgs, targets):
            print("Img tensor shape:", img_patch.shape)
            print("Boxes:", target["boxes"])
            visualize_patch(img_patch, target)
        break
