import os
import random as rnd

import cv2
import numpy as np
import torch
from torchvision.io import read_image
from tqdm import tqdm


def calculate_mean_std(data_path, n_samples: int = 1000):
    """
    Calculates the mean and standard deviation of the dataset located at the specified path.
    Args:
        data_path (str): The file path to the dataset. The dataset should be in a format
                         that can be loaded and processed for statistical calculations.
        n_samples (str): How many samples to use for the mean and std calculation.
    Returns:
        tuple: A tuple containing two elements:
               - mean (float): The mean value of the dataset.
               - std (float): The standard deviation of the dataset.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the dataset is empty or cannot be processed.
    """
    input_dir = os.path.join(data_path, "target")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_names = os.listdir(input_dir)
    if not image_names:
        raise ValueError("Dataset is empty")

    rnd.shuffle(image_names)
    image_names = image_names[:n_samples]

    mean = torch.zeros(3)
    m2 = torch.zeros(3)
    n_pixels = 0
    for image_name in tqdm(image_names, desc="Calculating online mean and std"):
        img = read_image(os.path.join(input_dir, image_name)).float() / 255.0
        pixels = img.numel() // 3
        n_pixels += pixels

        img_sum = img.sum(dim=[1, 2])
        delta = img_sum / pixels - mean
        mean += delta * (pixels / n_pixels)
        m2 += (img**2).sum(dim=[1, 2])

    variance = m2 / n_pixels - mean**2
    std = torch.sqrt(variance)
    return mean, std


def split_image(image: np.ndarray, patch_size=256):
    """
    Split an image into patches of a specified size.

    Parameters:
    - image: Input image (numpy array).
    - patch_size: Size of each patch (int).

    Returns:
    - List of patches (list of numpy arrays).
    """
    h, w = image.shape[:2]
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    image = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=0)
    patches = []

    for i in range(0, new_h, patch_size):
        for j in range(0, new_w, patch_size):
            patch = image[i : i + patch_size, j : j + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)

    return patches
