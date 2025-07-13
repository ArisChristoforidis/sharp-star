from typing import Annotated, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import typer
from PIL import Image
from predict import predict

app = typer.Typer()


@app.command()
def vizualize(
    input_path: Annotated[str, typer.Option("--input", "-i")],
    ground_truth_path: Annotated[str | None, typer.Option("--ground", "-g")],
    model_path: Annotated[str, typer.Option("--model", "-m")] = "models/model.pth",
) -> None:
    """
    Visualizes the input image, the model's predicted output, and optionally the ground truth image side by side.
    Args:
        input_path (str): Path to the input image file.
        ground_truth_path (str | None): Path to the ground truth image file. If None, ground truth is not displayed.
        model_path (str): Path to the trained model file.
    Displays:
        A matplotlib figure showing the input image, predicted output, and optionally the ground truth image.
    """

    predicted_tensor = predict(input_path, None, model_path)
    input_image = np.array(Image.open(input_path))
    prediction_image = predicted_tensor.permute(1, 2, 0).numpy()

    image_titles = ["Input", "Predicted"]
    images = [input_image, prediction_image]
    if ground_truth_path:
        ground_truth_image = np.array(Image.open(ground_truth_path))
        images.append(ground_truth_image)
        image_titles.append("Ground Truth")

    _, axes = plt.subplots(1, len(images))
    for axis, img, title in zip(axes, images, image_titles):
        axis.imshow(img)
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    vizualize()
