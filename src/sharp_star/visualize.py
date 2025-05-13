from typing import Annotated, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import typer
from PIL import Image
from predict import predict

app = typer.Typer()


@app.command()
def vizualize(
    input_path: Annotated[str, typer.Option("--input", "-i")] = "data/splits/test/input/0.jpg",
    ground_truth_path: Annotated[str | None, typer.Option("--ground", "-g")] = "data/splits/test/target/0.jpg",
    model_path: Annotated[str, typer.Option("--model", "-m")] = "models/model.pth",
) -> None:
    predicted_tensor = predict(input_path, None, model_path)
    input_image = np.array(Image.open(input_path))
    prediction_image = predicted_tensor.permute(1, 2, 0).numpy()

    image_titles = ["Input", "Predicted"]
    images = [input_image, prediction_image]
    if ground_truth_path:
        ground_truth_image = np.array(Image.open(ground_truth_path))
        images.append(ground_truth_image)
        image_titles.append("Ground Truth")

    _, axes = plt.subplots(1, 3)
    for axis, img, title in zip(axes, images, image_titles):
        axis.imshow(img)
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    vizualize()
