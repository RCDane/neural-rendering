import argparse
import numpy as np
from PIL import Image
import flip_evaluator as flip

def calculate_flip_metric(image1_path, image2_path):
    """
    Calculates the FLIP metric between two images.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.

    Returns:
        float: The FLIP metric value.
    """
    # Load images
    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")

    # Convert to numpy arrays
    img1_np = np.array(img1, dtype=np.float32) / 255.0
    img2_np = np.array(img2, dtype=np.float32) / 255.0

    # Calculate FLIP
    flip_map, flip_mean, dict = flip.evaluate(img1_np, img2_np, "LDR")

    return flip_map, flip_mean

import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the NVIDIA FLIP metric between two images.")
    parser.add_argument("image1", type=str, help="Path to the first PNG image.")
    parser.add_argument("image2", type=str, help="Path to the second PNG image.")
    parser.add_argument("--output", type=str, default="flip_result.png", help="Path to save the FLIP distance map image.")
    args = parser.parse_args()

    flip_map, flip_mean = calculate_flip_metric(args.image1, args.image2)
    print(f"FLIP mean distance: {flip_mean}")
    
    plt.imshow(flip_map, cmap='hot')
    plt.colorbar()
    plt.title('FLIP Distance Map')
    plt.savefig(args.output)
    plt.show()