import torch
from sharp_star.data import AstroDataset
from sharp_star.denormalize import Denormalize
from torch.utils.data import Dataset
from torchvision import transforms


def test_normalization():
    """Test if normalization and denormalization work as intended."""
    # Dummy image tensor
    image = torch.rand(3, 256, 256)

    mean = torch.rand(3)
    std = torch.rand(3)

    normalize = transforms.Normalize(mean=mean, std=std)
    denormalize = Denormalize(mean=mean, std=std)

    normalized_image = normalize(image)
    denormalized_image = denormalize(normalized_image)

    assert torch.allclose(image, denormalized_image, atol=1e-6)
