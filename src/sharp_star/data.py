import os
from typing import Tuple

import torch
import typer
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class AstroDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: str, transform=None) -> None:
        self.data_path = data_path
        self.input_path = os.path.join(self.data_path, "input")
        self.target_path = os.path.join(self.data_path, "target")
        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        # TODO: Remove the limiter.
        return len(os.listdir(self.input_path)[:100])

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        input_path = os.path.join(self.input_path, f"{index}.jpg")
        target_path = os.path.join(self.target_path, f"{index}.jpg")
        input_image = read_image(input_path) / 255.0
        target_image = read_image(target_path) / 255.0
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

    def __str__(self):
        return f"AstroDataset with {len(self)} samples. ({self.data_path})"


def make_dataset(path: str, mean: torch.Tensor, std: torch.Tensor) -> AstroDataset:
    """
    Creates and returns training and evaluation datasets.
    Args:
        path (str): The file path to the dataset.
        mean (torch.Tensor): The mean for normalization.
        std (torch.Tensor): The standard deviation for normalization.
    Returns:
        AstroDataset: The dataset.
    """
    normalize = transforms.Normalize(mean=mean, std=std)
    dataset = AstroDataset(path, transform=normalize)
    return dataset


if __name__ == "__main__":
    typer.run(make_dataset)
