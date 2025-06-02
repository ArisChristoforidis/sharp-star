import os
from typing import Annotated

import torch
import typer
from evaluate import evaluate
from model import UNet
from torch.nn import L1Loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import calculate_mean_std

import wandb
from data import make_dataset

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@app.command()
def train(
    train_path: Annotated[str, typer.Option("--train", "-t")],
    eval_path: Annotated[str, typer.Option("--eval", "-v")],
    output_path: Annotated[str, typer.Option("--output", "-o")],
    checkpoint: Annotated[str, typer.Option("--checkpoint", "-c")] = None,
    lr: Annotated[float, typer.Option("--learning_rate", "-lr")] = 1e-3,
    batch_size: Annotated[int, typer.Option("--batch", "-b")] = 32,
    epochs: Annotated[int, typer.Option("--epochs", "-e")] = 10,
    log_metrics: Annotated[bool, typer.Option("--log", "-l")] = True,
    autocast: Annotated[bool, typer.Option("--autocast", "-a")] = False,
) -> None:
    """
    Trains a UNet model for image-to-image tasks with optional checkpoint resumption and metric logging.
    Args:
        train_path (str): Path to the training dataset.
        eval_path (str): Path to the evaluation dataset.
        output_path (str): Path to save the trained model checkpoint.
        checkpoint (str, optional): Path to a checkpoint file to resume training from. Defaults to None
        lr (float): Learning rate for the optimizer. Defaults to 1e-5.
        batch_size (int): Batch size for training. Defaults to 32.
        epochs (int): Number of training epochs. Defaults to 10.
        log_metrics (bool): Whether to log metrics using Weights & Biases (wandb). Defaults to True.
        autocast (bool): Whether to use mixed precision with autocast or not. Defaults to False
    """
    original_model = UNet(in_channels=3, out_channels=3)
    optimizer = Adam(original_model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5)
    scaler = torch.GradScaler(device=str(DEVICE))
    l1_loss = L1Loss()

    if checkpoint:
        checkpoint_data = torch.load(checkpoint, map_location="cpu")

        original_model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint_data["scaler_state_dict"])

        mean = torch.tensor(checkpoint_data["mean"])
        std = torch.tensor(checkpoint_data["std"])

        start_epoch = checkpoint_data["epoch"] + 1
        wandb_id = checkpoint_data["wandb_id"]
        print(f"Resuming from checkpoint {checkpoint} [{start_epoch} Epoch(s)]")
    else:
        start_epoch = 0
        mean, std = calculate_mean_std(train_path)
        wandb_id = None

    original_model = original_model.to(DEVICE)
    compiled_model = torch.compile(original_model)

    # Check if optimizer_state_dict was actually loaded.
    if checkpoint:
        print(f"Moving optimizer state to {DEVICE}...")
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
        print(f"Optimizer state successfully moved to {DEVICE}.")

    train_set = make_dataset(train_path, mean, std)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4)

    if log_metrics:
        metrics = wandb.init(
            project="sharp_star",
            resume="allow",
            id=wandb_id,
            config={"architecture": "UNet", "lr": lr, "batch_size": batch_size, "epochs": epochs},
        )

    for epoch in range(start_epoch, start_epoch + epochs):
        original_model.train()
        for i, (input_image, target_image) in tqdm(
            enumerate(train_loader), desc=f"Training epoch {epoch + 1}", total=len(train_loader)
        ):
            input_image, target_image = input_image.to(DEVICE), target_image.to(DEVICE)
            optimizer.zero_grad()
            # Forward pass
            with torch.autocast(device_type=str(DEVICE), dtype=torch.float16, enabled=autocast):
                pred_image = compiled_model(input_image)
                loss = l1_loss(pred_image, target_image)

            # Back pass
            if autocast:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(original_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        new_checkpoint = {
            "model_state_dict": original_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
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
        scheduler.step(eval_l1)

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
    train(train_path="data/reduced/train", eval_path="data/reduced/eval", output_path="models/model.pth")
