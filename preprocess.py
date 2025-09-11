import cv2
import numpy as np
import torch


def preprocess_image(img, apply_hough=True):
    """
    Preprocesses an image for training:
      1. Convert to grayscale
      2. Binarize (adaptive threshold)
      3. Optionally overlay Hough lines
      4. Convert to Torch tensor (C,H,W) normalized to [0,1]
    """
    # --- Step 1: grayscale ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Step 2: binarization ---
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # --- Step 3: Hough transform (optional) ---
    if apply_hough:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )
        if lines is not None:
            # Convert binary to 3 channels for drawing
            binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.line(binary_color, (x1, y1), (x2, y2), color, 2)
        else:
            # If no lines detected, just use grayscale binary 3ch
            binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    else:
        binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # --- Step 4: convert to tensor ---
    tensor = torch.from_numpy(binary_color).permute(2, 0, 1).float() / 255.0
    return tensor


def preprocess_image_for_symbol(img):
    """
    Preprocessing for symbols:
      - Grayscale
      - Adaptive threshold (binarization)
      - Convert to tensor
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    tensor = torch.from_numpy(binary).unsqueeze(0).float() / 255.0  # (1,H,W)
    return tensor


def preprocess_image_for_word(img):
    """
    Preprocessing for words:
      - Grayscale
      - Normalize to [0,1]
      - Optional light binarization (global threshold)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    tensor = torch.from_numpy(binary).unsqueeze(0).float() / 255.0  # (1,H,W)
    return tensor


def preprocess_image_for_line(img, apply_hough=True):
    """
    Preprocessing for lines:
      - Grayscale
      - Adaptive threshold
      - Optionally overlay Hough lines
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )

    if apply_hough:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )
        binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                color = (0, 255, 0)  # green lines
                cv2.line(binary_color, (x1, y1), (x2, y2), color, 2)
        tensor = torch.from_numpy(binary_color).permute(2, 0, 1).float() / 255.0
    else:
        tensor = torch.from_numpy(binary).unsqueeze(0).float() / 255.0

    return tensor
