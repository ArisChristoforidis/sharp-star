import typer
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import os
from utils import calculate_mean_std

class AstroDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: str, transform=None) -> None:
        self.data_path = data_path
        self.input_path = os.path.join(self.data_path, 'input')
        self.target_path = os.path.join(self.data_path, 'target')
        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(os.listdir(self.input_path))

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

    def preprocess(self, output_folder: str) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(data_path: str, output_folder: str) -> None:
    print("Preprocessing data...")
    mean, std = calculate_mean_std(data_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset = AstroDataset(data_path, transform=transform)
    dataset.preprocess(output_folder)

if __name__ == "__main__":
    typer.run(preprocess)
