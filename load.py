import json
from pathlib import Path

import cv2
import numpy as np

dataset_dir = Path("dataset")
images_dir = Path("images")
output_dir = Path("annotations")
output_dir.mkdir(exist_ok=True)


def parse_symbols(path, objects):
    symbols = np.load(path, allow_pickle=True)
    for s in symbols:
        _, bbox, cls_id = s
        x1, y1, x2, y2 = bbox
        objects.append({
            "category": "symbol",
            "class_id": int(cls_id),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })


def parse_words(path, objects):
    words = np.load(path, allow_pickle=True)
    for w in words:
        _, bbox, text, *_ = w
        x1, y1, x2, y2 = bbox
        objects.append({
            "category": "word",
            "text": str(text),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })


def parse_lines(path, objects):
    lines = np.load(path, allow_pickle=True)
    for l in lines:
        _, coords, *_ = l
        x1, y1, x2, y2 = coords
        objects.append({
            "category": "line",
            "bbox": [
                int(min(x1, x2)),
                int(min(y1, y2)),
                int(max(x1, x2)),
                int(max(y1, y2)),
            ],
        })


def parse_lines2(path, objects):
    # we can ignore lines2 if not needed
    lines2 = np.load(path, allow_pickle=True)
    for l in lines2:
        x1, y1, x2, y2, *_ = l
        objects.append({
            "category": "line2",
            "bbox": [
                int(min(x1, x2)),
                int(min(y1, y2)),
                int(max(x1, x2)),
                int(max(y1, y2)),
            ],
        })


def parse_linker(path, relations):
    linkers = np.load(path, allow_pickle=True)
    relations["linker"] = []
    for l in linkers:
        src, targets = l
        relations["linker"].append([str(src), list(map(str, targets))])


def parse_table(path, relations):
    table = np.load(path, allow_pickle=True)
    relations["table"] = table.tolist()


def parse_keyvalue(path, relations):
    kv = np.load(path, allow_pickle=True)
    relations["keyvalue"] = kv.tolist()


def parse(base, idx, ann):
    parse_symbols(base / f"{idx}_symbols.npy", ann["objects"])
    parse_words(base / f"{idx}_words.npy", ann["objects"])
    parse_lines(base / f"{idx}_lines.npy", ann["objects"])
    parse_lines2(base / f"{idx}_lines2.npy", ann["objects"])
    parse_linker(base / f"{idx}_linker.npy", ann["relations"])
    parse_table(base / f"{idx}_Table.npy", ann["relations"])
    parse_keyvalue(base / f"{idx}_KeyValue.npy", ann["relations"])


# --------- main loop ----------
for idx in range(500):  # adjust to dataset size
    print(idx, end="---")

    img_path = images_dir / f"{idx}.jpg"
    if not img_path.exists():
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    ann = {
        "image_id": str(idx),
        "width": w,
        "height": h,
        "objects": [],
        "relations": {},
    }

    base = dataset_dir / str(idx)

    parse(base, idx, ann)

    with open(output_dir / f"{idx}.json", "w") as f:
        json.dump(ann, f, indent=2)
