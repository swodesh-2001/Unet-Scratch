import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from models import UNET
from utils import load_checkpoint
import cv2
import argparse


IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240


def create_green_overlay(image, mask, alpha=0.3):
    """Create a green overlay mask on top of the original image."""
    green_mask = np.zeros_like(image)
    green_mask[:, :, 1] = (mask * 255).squeeze()  # Green channel

    # Combine original image with the green mask
    overlay = cv2.addWeighted(image, 1, green_mask, alpha, 0)
    return overlay


def single_image_inference(image_path, model, device):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        image = np.array(Image.open(image_path).convert("RGB"))

        inference_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        )

        img = inference_transform(image=image)["image"].unsqueeze(0).to(device)
        pred_mask = model(img)

        img = img.squeeze(0).cpu().permute(1, 2, 0).numpy()
        pred_mask = pred_mask.squeeze(0).cpu().permute(1, 2, 0).numpy()
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1

        overlay = create_green_overlay(img,pred_mask)


        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[1].imshow(pred_mask, cmap="gray")
        axes[1].set_title("Predicted Mask")
        axes[2].imshow(overlay)
        axes[2].set_title("OVerlay image")
        plt.show()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Run inference on a single image using a UNET model")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()

    SINGLE_IMG_PATH = args.image_path
    MODEL_PATH = args.model_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNET(in_channels=3, out_channels=1).to(device=device)
    load_checkpoint(MODEL_PATH, model=model)
    single_image_inference(SINGLE_IMG_PATH, model, device)
