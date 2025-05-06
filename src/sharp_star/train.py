import os
from typing import Annotated

import torch
import typer
import wandb
from model import UNet
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import calculate_mean_std

from data import make_dataset

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@app.command()
def train(
        data_path: Annotated[str, typer.Option('--data', '-d')] = "data/splits/train",
        output_path: Annotated[str, typer.Option('--output', '-o')] = "models/model.pth",
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 1,
        log_metrics: bool = False
    ) -> None:

    mean, std = calculate_mean_std(data_path)
    train_set = make_dataset(data_path, mean, std)

    train_loader = DataLoader(train_set, batch_size=batch_size)

    model = UNet(in_channels=3, out_channels=3)
    optimizer = Adam(model.parameters(), lr=lr)
    l1_loss = L1Loss()

    if log_metrics:
        metrics = wandb.init(
            project="sharp_star", config={"architecture": "UNet", "lr": lr, "batch_size": batch_size, "epochs": epochs}
        )

    for epoch in range(epochs):
        model.train()
        for i, (input_image, target_image) in tqdm(enumerate(train_loader), desc=f'Training epoch {epoch+1}'):
            input_image, target_image = input_image.to(DEVICE), target_image.to(DEVICE)
            optimizer.zero_grad()
            # Forward pass
            pred_image = model(input_image)
            loss = l1_loss(pred_image, target_image)
            # Back pass
            loss.backward()
            optimizer.step()

            if log_metrics:
                metrics.log({"train_loss": loss.item()})

    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': mean.tolist(),
        'std': std.tolist()
        },
        output_path
    )

if __name__ == "__main__":
    train()
