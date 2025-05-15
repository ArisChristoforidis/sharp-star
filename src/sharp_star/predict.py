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
    """
    Reconstructs an image from a list of patches.
    Args:
        original_shape (Tuple[int, int]): The shape of the original image as (channels, height, width).
        patches (List[torch.Tensor]): A list of image patches as tensors, each of shape (channels, patch_size, patch_size).
        patch_size (int): The size of each square patch.
    Returns:
        torch.Tensor: The reconstructed image tensor of shape (channels, height, width), cropped to the original size.
    """

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
    image_path: Annotated[str, typer.Option("--input", "-i")],
    output_path: Annotated[str | None, typer.Option("--output", "-o")],
    model_path: Annotated[str, typer.Option("--model", "-m")],
    batch_size: Annotated[int, typer.Option("--batch", "-b")] = 8,
    patch_size: Annotated[int, typer.Option("--patch", "-p")] = 256,
) -> None:
    """
    Runs inference on an input image using a pre-trained UNet model, processes the image in patches, and optionally saves the output.
    Args:
        image_path (str): Path to the input image file.
        output_path (str | None): Path to save the output image. If None, the output is not saved.
        model_path (str): Path to the trained model checkpoint (.pth file).
        batch_size (int): Number of patches to process in a batch during inference.
        patch_size (int): Size of the square patches to split the image into.
    Returns:
        torch.Tensor: The output image tensor after model inference and denormalization.
    """
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
