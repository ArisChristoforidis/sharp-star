from sharp_star.model import UNet
import torch

def test_model():
    x = torch.randn(1, 3, 256, 256)
    model = UNet(in_channels=3, out_channels=3)
    y = model(x)
    assert y.shape == (1, 3, 256, 256)