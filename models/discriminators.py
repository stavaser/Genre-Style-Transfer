"""This file contains all discriminators."""
from numpy import pad
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, stride, 3, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class ResDiscriminator(nn.Module):
    def __init__(self, k_count, input_ch_size, noise:bool=True):
        super().__init__()
        self.noise = noise

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=input_ch_size,
                      out_channels=k_count,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False,
                      padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = k_count

        layers.append(Block(in_channels=in_channels,
                            out_channels=in_channels*4,
                            stride=2))
        in_channels = in_channels*4


        layers.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=1,
                                kernel_size=7,
                                padding=3,
                                bias=False,
                                stride=1,
                                padding_mode='reflect'))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.noise:
            noise = (0.18**2)*torch.randn(x.shape).cuda()
            out = x + noise
            out = self.initial(out)
            del noise
        else:
            out = self.initial(x)
        return torch.sigmoid(self.model(out))
