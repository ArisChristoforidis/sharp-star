from typing import Annotated

import numpy as np
import torch
import typer
from denormalize import Denormalize
from model import UNet
from torchvision import transforms
from torchvision.io import read_image
from utils import split_image

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def get_image_patches(image: torch.tensor):
    np_image = image.numpy()
    # Split into patches.
    patches = [torch.from_numpy(np.transpose(patch, (2, 0, 1))) for patch in split_image(np_image)]
    return patches


@app.command()
def predict(
    image_path: Annotated[str, typer.Option("--input", "-i")],
    output_path: Annotated[str, typer.Option("--output", "-o")] = "out.jpg",
    model_path: Annotated[str, typer.Option("--model", "-m")] = "models/model.pth",
    batch_size: Annotated[int, typer.Option("--batch", "-b")] = 8,
) -> None:
    checkpoint = torch.load(model_path, map_location="cpu")

    model = UNet(in_channels=3, out_channels=3)
    model.load_state_dict(checkpoint["model_state_dict"])

    mean = torch.tensor(checkpoint["mean"])
    std = torch.tensor(checkpoint["std"])

    model.eval()
    model.to(DEVICE)

    input_transform = transforms.Normalize(mean=mean, std=std)
    output_transform = Denormalize(mean=mean, std=std)

    image = read_image(image_path) / 255.0
    image = input_transform(image)
    image_patches = get_image_patches(image)

    for i in range(0, len(image_patches), batch_size):
        batch = torch.stack(image_patches[i : i + batch_size]).to(DEVICE)
        with torch.no_grad():
            out = model(batch)
        pass
