import os
from typing import Annotated

import torch
import typer
from evaluate import evaluate
from model import UNet
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import calculate_mean_std

import wandb
from data import make_dataset

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@app.command()
def train(
    checkpoint: Annotated[str, typer.Option("--checkpoint", "-c")] = None,
    train_path: Annotated[str, typer.Option("--train", "-t")] = "data/reduced/train",
    eval_path: Annotated[str, typer.Option("--eval", "-e")] = "data/reduced/eval",
    output_path: Annotated[str, typer.Option("--output", "-o")] = "models/model.pth",
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 10,
    log_metrics: bool = True,
) -> None:
    model = UNet(in_channels=3, out_channels=3)
    optimizer = Adam(model.parameters(), lr=lr)
    l1_loss = L1Loss()

    if checkpoint:
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model = model.to(DEVICE)
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        mean = torch.tensor(checkpoint_data["mean"])
        std = torch.tensor(checkpoint_data["std"])
        start_epoch = checkpoint_data["epoch"]
        wandb_id = checkpoint_data["wandb_id"]
        print(f"Resuming from checkpoint {checkpoint} [{start_epoch} Epoch(s)]")
    else:
        start_epoch = 0
        mean, std = calculate_mean_std(train_path)
        wandb_id = None
        model = model.to(DEVICE)

    train_set = make_dataset(train_path, mean, std)
    train_loader = DataLoader(train_set, batch_size=batch_size)

    if log_metrics:
        metrics = wandb.init(
            project="sharp_star",
            resume="allow",
            id=wandb_id,
            config={"architecture": "UNet", "lr": lr, "batch_size": batch_size, "epochs": epochs},
        )

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        for i, (input_image, target_image) in tqdm(
            enumerate(train_loader), desc=f"Training epoch {epoch+1}", total=len(train_loader)
        ):
            input_image, target_image = input_image.to(DEVICE), target_image.to(DEVICE)
            optimizer.zero_grad()
            # Forward pass
            pred_image = model(input_image)
            loss = l1_loss(pred_image, target_image)
            # Back pass
            loss.backward()
            optimizer.step()

        new_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

        if log_metrics:
            new_checkpoint["wandb_id"] = metrics.id

        torch.save(
            new_checkpoint,
            output_path,
        )
        # Evaluate and log.
        eval_l1, psnr, ssim = evaluate(model_path=output_path, eval_path=eval_path, verbose=False)

        if log_metrics:
            metrics.log(
                {
                    "Train Loss": loss.item(),
                    "Validation Loss": eval_l1,
                    "Validation PSNR": psnr,
                    "Validation SSIM": ssim,
                }
            )


if __name__ == "__main__":
    train()
