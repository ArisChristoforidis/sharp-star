import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

from sharp_star.denormalize import Denormalize
from sharp_star.data import AstroDataset
from sharp_star.utils import inverse_normalization
import random as rnd
from tests import _PATH_DATA

def test_my_dataset():
    """Test the MyDataset class."""
    dataset = AstroDataset("data/raw")
    assert isinstance(dataset, Dataset)

def test_normalization():
    """ Test if normalization and denormalization work as intended."""
    # Dummy image tensor
    image = torch.rand(3, 256, 256)
    
    mean = torch.rand(3)
    std = torch.rand(3)

    normalize = transforms.Normalize(mean=mean, std=std)
    denormalize = Denormalize(mean=mean, std=std)
    
    normalized_image = normalize(image)
    denormalized_image = denormalize(normalized_image)

    assert torch.allclose(image, denormalized_image, atol=1e-6)
