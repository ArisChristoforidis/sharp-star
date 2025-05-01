import torch


class Denormalize(object):
    """
    A class to reverse the normalization of image tensors by applying the
    mean and standard deviation used during normalization.
    Attributes:
        mean (torch.Tensor): A tensor containing the mean values for each channel.
                             It is reshaped to have dimensions (3, 1, 1) to match
                             the shape of the input tensor.
        std (torch.Tensor): A tensor containing the standard deviation values for
                            each channel. It is reshaped to have dimensions (3, 1, 1).
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.view(3, 1, 1)
        self.std = std.view(3, 1, 1)

    def __call__(self, sample):
        return sample * self.std + self.mean
