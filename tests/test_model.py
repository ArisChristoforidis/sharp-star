import torch
from sharp_star.model import Generator


def test_model():
    x = torch.randn(1, 3, 256, 256)
    model = Generator(in_channels=3, out_channels=3)
    y = model(x)
    assert y.shape == (1, 3, 256, 256)
