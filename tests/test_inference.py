import math

import pytest
import torch
from sharp_star.utils import split_image


@pytest.mark.parametrize("image_size", [256, 500, 1024])
def test_split_image(image_size: int, patch_size: int = 256):
    image = torch.rand(3, image_size, image_size)

    patches = split_image(image)
    assert (
        len(patches) == (math.ceil(image_size / patch_size)) ** 2
    ), f"Expected {(math.ceil(image_size / patch_size)) ** 2} patches, but got {len(patches)}"
