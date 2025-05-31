import os
from typing import Annotated

import torch
import typer
from evaluate import evaluate
from model import Discriminator, Generator
from torch.nn import BCELoss, L1Loss
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
    generator_checkpoint: Annotated[str, typer.Option("--generator", "-g")] = None,
    discriminator_checkpoint: Annotated[str, typer.Option("--discriminator", "-d")] = None,
    lr: Annotated[float, typer.Option("--learning_rate", "-lr")] = 1e-5,
    batch_size: Annotated[int, typer.Option("--batch", "-b")] = 32,
    epochs: Annotated[int, typer.Option("--epochs", "-e")] = 10,
    log_metrics: Annotated[bool, typer.Option("--log", "-l")] = True,
) -> None:
    """
    Trains a UNet model for image-to-image tasks with optional checkpoint resumption and metric logging.
    Args:
        train_path (str): Path to the training dataset.
        eval_path (str): Path to the evaluation dataset.
        output_path (str): Path to save the trained model checkpoint.
        generator_checkpoint (str, optional): Path to the generator checkpoint file to resume training from.
        discriminator_checkpoint (str, optional): Path to the discriminator checkpoint file to resume training from.
        lr (float): Learning rate for the optimizer. Defaults to 1e-5.
        batch_size (int): Batch size for training. Defaults to 32.
        epochs (int): Number of training epochs. Defaults to 10.
        log_metrics (bool): Whether to log metrics using Weights & Biases (wandb). Defaults to True.
    """
    assert (generator_checkpoint and discriminator_checkpoint) or not (
        generator_checkpoint and discriminator_checkpoint
    ), "Either both or none checkpoints should be available!"

    generator = Generator(in_channels=3, out_channels=3)
    discriminator = Discriminator(in_channels=3)
    optim_g = Adam(generator.parameters(), lr=lr)
    optim_d = Adam(discriminator.parameters(), lr=lr)
    scheduler_g = ReduceLROnPlateau(optim_g, "min", patience=5)
    scheduler_d = ReduceLROnPlateau(optim_d, "min", patience=5)
    scaler = torch.GradScaler(device=str(DEVICE))
    l1_loss = L1Loss()
    bce_loss = BCELoss()

    if generator_checkpoint and discriminator_checkpoint:
        generator_data = torch.load(generator_checkpoint, map_location="cpu")
        discriminator_data = torch.load(discriminator_checkpoint, map_location="cpu")

        # Models
        generator.load_state_dict(generator_data["model_state_dict"])
        discriminator.load_state_dict(discriminator_data["model_state_dict"])

        # Optimizers
        optim_g.load_state_dict(generator_data["optimizer_state_dict"])
        optim_d.load_state_dict(discriminator_data["optimizer_state_dict"])

        # Schedulers
        scheduler_g.load_state_dict(generator_data["scheduler_state_dict"])
        scheduler_d.load_state_dict(generator_data["scheduler_state_dict"])

        # TODO
        scaler.load_state_dict(generator_data["scaler_state_dict"])

        mean = torch.tensor(generator_data["mean"])
        std = torch.tensor(generator_data["std"])

        start_epoch = generator_data["epoch"] + 1
        wandb_id = generator_data["wandb_id"]
        print(f"Resuming from checkpoint {generator_checkpoint}|{discriminator_checkpoint} [{start_epoch} Epoch(s)]")
    else:
        start_epoch = 0
        mean, std = calculate_mean_std(train_path)
        wandb_id = None

    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)
    compiled_generator = torch.compile(generator)
    compiled_discriminator = torch.compile(discriminator)

    # Check if optimizer_state_dict was actually loaded.
    if generator_checkpoint and discriminator_checkpoint:
        print(f"Moving optimizer states to {DEVICE}...")
        for state in optim_g.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)

        for state in optim_d.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
        print(f"Optimizer states successfully moved to {DEVICE}.")

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
        generator.train()
        discriminator.train()
        for i, (input_image, target_image) in tqdm(
            enumerate(train_loader), desc=f"Training epoch {epoch + 1}", total=len(train_loader)
        ):
            input_image, target_image = input_image.to(DEVICE), target_image.to(DEVICE)
            optim_g.zero_grad()
            optim_d.zero_grad()
            # Discriminator forward pass
            with torch.autocast(device_type=str(DEVICE), dtype=torch.float16):
                g_out = compiled_generator(input_image)
                d_real = compiled_discriminator(input_image, input_image)
                loss_real = bce_loss(d_real, torch.ones_like(d_real))
                d_fake = compiled_discriminator(input_image, g_out)
                loss_fake = bce_loss(d_fake, torch.zeros_like(d_fake))
                d_loss = (loss_real + loss_fake) / 2

            # Discriminator back pass
            scaler.scale(d_loss).backward()
            scaler.unscale_(optim_d)
            # TODO: Remove this?
            clip_grad_norm_(discriminator.parameters(), 1.0)
            scaler.step(optim_d)

            # Generator forward pass
            with torch.autocast(device_type=str(DEVICE), dtype=torch.float16):
                d_fake = compiled_discriminator(input_image, g_out)
                g_loss = bce_loss(d_fake, torch.ones_like(d_fake)) + l1_loss(g_out, input_image)

            # Generator back pass
            scaler.scale(g_loss).backward()
            scaler.unscale_(optim_g)
            # TODO: Remove this?
            clip_grad_norm_(generator.parameters(), 1.0)
            scaler.step(optim_g)

            # Run this once for the entire model!
            scaler.update()

        new_g_checkpoint = {
            "model_state_dict": generator.state_dict(),
            "optimizer_state_dict": optim_g.state_dict(),
            "scheduler_state_dict": scheduler_g.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

        new_d_checkpoint = {
            "model_state_dict": discriminator.state_dict(),
            "optimizer_state_dict": optim_d.state_dict(),
            "scheduler_state_dict": scheduler_d.state_dict(),
        }

        if log_metrics:
            new_g_checkpoint["wandb_id"] = metrics.id

        torch.save(
            new_g_checkpoint,
            os.join(output_path, "generator.pth"),
        )

        torch.save(
            new_d_checkpoint,
            os.join(output_path, "discriminator.pth"),
        )

        # Evaluate and log.
        eval_l1, psnr, ssim = evaluate(model_path=output_path, eval_path=eval_path, verbose=False)
        scheduler_d.step(eval_l1)
        scheduler_g.step(eval_l1)

        if log_metrics:
            metrics.log(
                {
                    "Generator Loss": g_loss.item(),
                    "Discriminator Loss": d_loss.item(),
                    "Validation Loss": eval_l1,
                    "Validation PSNR": psnr,
                    "Validation SSIM": ssim,
                }
            )


if __name__ == "__main__":
    train(
        train_path=os.path.join("data", "reduced", "train"),
        eval_path=os.path.join("data", "reduced", "eval"),
        output_path="models",
    )
