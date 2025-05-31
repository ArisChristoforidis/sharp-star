import torch
from torch import nn


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, bias: bool):
        super(ConvolutionBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            ConvolutionBlock(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            ConvolutionBlock(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, upscale_factor: int = 2):
        super(Up, self).__init__()
        self.up_conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), kernel_size=3, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.double_conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.up_conv(x)
        x = self.pixel_shuffle(x)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_features: int = 32):
        super(Generator, self).__init__()

        self.in_conv = DoubleConv(in_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)
        self.down4 = Down(base_features * 8, base_features * 16)

        self.up1 = Up(base_features * 16, base_features * 8, base_features * 8)
        self.up2 = Up(base_features * 8, base_features * 4, base_features * 4)
        self.up3 = Up(base_features * 4, base_features * 2, base_features * 2)
        self.up4 = Up(base_features * 2, base_features, base_features)
        self.out_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return logits


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, base_features: int = 32):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_features, kernel_size=3, padding=1),
            nn.ReLU(),
            ConvolutionBlock(base_features, base_features * 2, kernel_size=3, padding=1, bias=False),
            ConvolutionBlock(base_features * 2, base_features * 4, kernel_size=3, padding=1, bias=False),
            ConvolutionBlock(base_features * 4, base_features * 8, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(base_features * 8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


if __name__ == "__main__":
    generator_input = torch.rand(8, 3, 64, 64)
    generator = Generator(in_channels=3, out_channels=3)

    discriminator_input_1 = torch.rand(8, 3, 64, 64)
    discriminator_input_2 = torch.rand(8, 3, 64, 64)
    discriminator = Discriminator(in_channels=3)

    print(f"Generator Parameters: {sum(p.numel() for p in generator.parameters() if p.requires_grad)}")
    print(f"[Generator] Input shape: {generator_input.shape}")
    print(f"[Generator] Output shape: {generator(generator_input).shape}")

    print(f"Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad)}")
    print(f"[Discriminator] Input 1 shape: {discriminator_input_1.shape}")
    print(f"[Discriminator] Input 2 shape: {discriminator_input_2.shape}")
    print(f"[Discriminator] Output shape: {discriminator(discriminator_input_1, discriminator_input_2).shape}")
