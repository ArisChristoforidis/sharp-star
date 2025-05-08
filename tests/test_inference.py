import math

import pytest
import torch
from sharp_star.predict import get_image_patches


@pytest.mark.parametrize("image_size", [256, 500, 1024])
def test_split_image(image_size: int, patch_size: int = 256):
    image = torch.rand(image_size, image_size, 3)

    patches = get_image_patches(image)
    assert (
        len(patches) == (math.ceil(image_size / patch_size)) ** 2
    ), f"Expected {(image_size // patch_size) ** 2} patches, but got {len(patches)}"
