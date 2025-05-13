import math
from typing import Annotated, List, Tuple

import torch
import typer
from denormalize import Denormalize
from model import UNet
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from utils import split_image

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def stitch_image(original_shape: Tuple[int, int], patches: List[torch.tensor], patch_size: int):
    channels, height, width = original_shape

    patches = torch.stack(patches)
    h_chunks = (height + patch_size - 1) // patch_size
    w_chunks = (width + patch_size - 1) // patch_size

    x = patches.reshape(h_chunks, w_chunks, channels, patch_size, patch_size).permute(2, 0, 3, 1, 4)
    x = x.reshape(channels, h_chunks * patch_size, w_chunks * patch_size)

    # Crop back to original size
    return x[:, :height, :width]


@app.command()
def predict(
    image_path: Annotated[str, typer.Option("--input", "-i")] = "data/splits/test/input/36.jpg",
    output_path: Annotated[str | None, typer.Option("--output", "-o")] = None,
    model_path: Annotated[str, typer.Option("--model", "-m")] = "models/model.pth",
    batch_size: Annotated[int, typer.Option("--batch", "-b")] = 8,
    patch_size: Annotated[int, typer.Option("--patch", "-p")] = 256,
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
    image_patches = split_image(image, patch_size)

    output_patches = []
    for i in tqdm(range(0, len(image_patches), batch_size)):
        batch = torch.stack(image_patches[i : i + batch_size]).to(DEVICE)
        with torch.no_grad():
            out = model(batch)

        output_patches.extend(out.unbind(0))
    out_image = stitch_image(image.shape, output_patches, patch_size)

    out_image = output_transform(out_image)
    if output_path:
        image_to_save = (out_image * 255).clamp(0, 255).byte()
        pil_image = to_pil_image(image_to_save)
        pil_image.save(output_path)

    return out_image


if __name__ == "__main__":
    predict()
