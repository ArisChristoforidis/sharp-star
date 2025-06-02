import os
from typing import List

import torch
import typer
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class AstroDataset(Dataset):
    """An Astrophotography image to image dataset.
    Args:
        data_path (str): Path to the data. Should contain 'input' and 'target' subdirectories.
        transform (List[torch.nn.Module] | None): A list of transforms to apply to the images.
    """

    def __init__(self, data_path: str, transform: List[torch.nn.Module] = None) -> None:
        self.data_path = data_path
        self.input_path = os.path.join(self.data_path, "input")
        self.target_path = os.path.join(self.data_path, "target")
        self.file_names = os.listdir(self.input_path)
        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(os.listdir(self.input_path))

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        file_name = self.file_names[index]
        input_path = os.path.join(self.input_path, file_name)
        target_path = os.path.join(self.target_path, file_name)
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
    dataset = AstroDataset(path, transform=None)
    return dataset


if __name__ == "__main__":
    typer.run(make_dataset)
