import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_and_save_prediction(
    dataset, idx, trainer, device, thickness=4, save_dir="outputs", name="symbol"
):
    """
    Visualize original and predicted bounding boxes stacked vertically,
    and save both images to disk.

    Args:
        dataset: PIDJSONDataset or similar
        idx: int, index of the sample in the dataset
        trainer: Trainer object containing the model
        device: torch device
        thickness: int, thickness of bounding boxes
        save_dir: directory to save the images
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    model = trainer.model
    model.eval()  # Set model to evaluation mode

    # Load image from path
    img_path = f"dataset/images/{dataset.ids[idx]}.jpg"
    original_image = cv2.imread(img_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Save original image
    orig_save_path = os.path.join(save_dir, f"{dataset.ids[idx]}_original.png")
    cv2.imwrite(orig_save_path, original_image)

    # Get preprocessed image tensor from dataset
    image_tensor, _ = dataset[idx]
    image_tensor = image_tensor.to(device)

    # Model inference
    with torch.no_grad():
        predictions = model([image_tensor])[0]

    # Convert tensor to numpy image for drawing boxes
    img_boxed = image_tensor.cpu().permute(1, 2, 0).numpy()
    img_boxed = (img_boxed * 255).astype(np.uint8)
    if img_boxed.shape[2] == 1:
        img_boxed = cv2.cvtColor(img_boxed, cv2.COLOR_GRAY2BGR)
    img_boxed = cv2.cvtColor(img_boxed, cv2.COLOR_RGB2BGR)

    # Draw red bounding boxes
    for box in predictions["boxes"]:
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(
            img_boxed, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=thickness
        )

    img_boxed_rgb = cv2.cvtColor(img_boxed, cv2.COLOR_BGR2RGB)

    # Save boxed image
    boxed_save_path = os.path.join(save_dir, f"{dataset.ids[idx]}_{name}.png")
    cv2.imwrite(boxed_save_path, cv2.cvtColor(img_boxed_rgb, cv2.COLOR_RGB2BGR))

    # Plot vertically
    plt.figure(figsize=(10, 14))
    plt.subplot(2, 1, 1)
    plt.imshow(original_image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.imshow(img_boxed_rgb)
    plt.title("Predicted Bounding Boxes")
    plt.axis("off")
    plt.show()

    print(f"Original image saved at: {orig_save_path}")
    print(f"Boxed image saved at: {boxed_save_path}")
