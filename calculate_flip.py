import argparse
import numpy as np
from PIL import Image
import flip_evaluator as flip
from pathlib import Path
try:
    from skimage.metrics import structural_similarity
except ImportError as exc:
    structural_similarity = None
    _ssim_import_error = exc
else:
    _ssim_import_error = None

def load_image_as_array(image_path):
    img = Image.open(image_path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0

def calculate_flip_metric(image1_path, image2_path, image1_np=None):
    """
    Calculates the FLIP metric between two images.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.

    Returns:
        tuple: (flip_map, flip_mean, reference_np, comparison_np).
    """
    if image1_np is None:
        image1_np = load_image_as_array(image1_path)
    image2_np = load_image_as_array(image2_path)
    flip_map, flip_mean, _ = flip.evaluate(image1_np, image2_np, "LDR")
    return flip_map, flip_mean, image1_np, image2_np

def calculate_ssim_metric(image1_np, image2_np):
    if structural_similarity is None:
        raise ImportError(
            "SSIM calculation requires scikit-image. Install it via `pip install scikit-image`."
        ) from _ssim_import_error
    try:
        return float(structural_similarity(image1_np, image2_np, channel_axis=2, data_range=1.0))
    except TypeError:
        return float(structural_similarity(image1_np, image2_np, multichannel=True, data_range=1.0))

def collect_image_paths(folder):
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for path in sorted(Path(folder).rglob("*")):
        if path.is_file() and path.suffix.lower() in valid_ext:
            yield path

import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the NVIDIA FLIP metric between two images.")
    parser.add_argument("--image1", type=str, help="Path to the first PNG image.")
    parser.add_argument("--image2", type=str, default="", help="Path to the second PNG image.")
    parser.add_argument("--folder", type=str, default="", help="Optional folder, with images inside.")
    parser.add_argument("--output", type=str, default="flip_result.png", help="Path to save the FLIP distance map image.")
    args = parser.parse_args()

    if args.folder:
        reference_np = load_image_as_array(args.image1)
        for image_path in collect_image_paths(args.folder):
            if str(image_path) == args.image1:
                continue
            flip_map, flip_mean, _, image2_np = calculate_flip_metric(
                args.image1, str(image_path), image1_np=reference_np
            )
            ssim_score = calculate_ssim_metric(reference_np, image2_np)
            print(f"{image_path}: FLIP={flip_mean:.6f}, SSIM={ssim_score:.6f}")
            output_path = image_path.with_name(f"{image_path.stem}_flip.png")
            plt.imsave(output_path, flip_map, cmap="hot")
        exit(0)
    flip_map, flip_mean, image1_np, image2_np = calculate_flip_metric(args.image1, args.image2)
    ssim_score = calculate_ssim_metric(image1_np, image2_np)
    print(f"FLIP mean distance: {flip_mean:.6f}")
    print(f"SSIM: {ssim_score:.6f}")
    plt.imshow(flip_map, cmap='hot')
    plt.savefig(args.output)
    plt.show()