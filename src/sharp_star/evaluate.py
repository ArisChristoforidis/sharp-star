from typing import Annotated

import torch
import torch.nn as nn
import typer
from model import UNet
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchvision import transforms
from tqdm import tqdm

import wandb
from data import make_dataset

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@app.command()
def evaluate(
    model_path: Annotated[str, typer.Option("--model", "-m")] = "models/model.pth",
    eval_path: Annotated[str, typer.Option("--data", "-d")] = "data/reduced/eval",
    batch_size: Annotated[int, typer.Option("--batch_size", "-b")] = 8,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = True,
) -> tuple[float, float, float]:
    """
    Evaluate a pre-trained model on a given evaluation dataset.
    Args:
        model_path (str): Path to the model checkpoint file. Defaults to 'models/model.pth'.
        eval_path (str): Path to the evaluation dataset. Defaults to "data/splits/eval".
        batch_size (int): Batch size for evaluation. Defaults to 8.
    Returns:
        tuple: A tuple containing:
            - average_l1 (float): Average L1 loss over the evaluation dataset.
            - average_psnr (float): Average Peak Signal-to-Noise Ratio (PSNR) over the evaluation dataset.
            - average_ssim (float): Average Structural Similarity Index Measure (SSIM) over the evaluation dataset.
    """

    checkpoint = torch.load(model_path, map_location="cpu")

    model = UNet(in_channels=3, out_channels=3)
    model.load_state_dict(checkpoint["model_state_dict"])

    mean = torch.tensor(checkpoint["mean"])
    std = torch.tensor(checkpoint["std"])
    eval_set = make_dataset(eval_path, mean, std)

    model.eval()
    model = torch.compile(model)
    model.to(DEVICE)

    psnr = PeakSignalNoiseRatio().to(DEVICE)
    ssim = StructuralSimilarityIndexMeasure().to(DEVICE)

    eval_loader = DataLoader(eval_set, batch_size=batch_size)
    total_l1, total_psnr, total_ssim = 0, 0, 0
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(eval_loader), desc=f"Evaluating model", total=len(eval_loader)):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            preds = model(inputs)

            loss_l1 = l1_loss(preds, targets)
            psnr_val = psnr(preds, targets)
            ssim_val = ssim(preds, targets)

            total_l1 += loss_l1.item()
            total_psnr += psnr_val.item()
            total_ssim += ssim_val.item()

        batch_count = len(eval_loader)
        average_l1 = total_l1 / batch_count
        average_psnr = total_psnr / batch_count
        average_ssim = total_ssim / batch_count

    if verbose:
        print(f"L1 Loss: {average_l1:.3f}")
        print(f"PSNR: {average_psnr:.3f}")
        print(f"SSIM: {average_ssim:.3f}")

    return average_l1, average_psnr, average_ssim


if __name__ == "__main__":
    evaluate()
