"""This file contains all generators."""

import torch
import torch.nn as nn
import sys
sys.path.insert(1, '../')

class Generator_ResNet(nn.Module):
    """Generator utilising resnet blocks.

    Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired Image-to-Image Translation using
    Cycle-Consistent Adversarial Networks. CoRR, abs/1703.10593. http://arxiv.org/abs/1703.10593

    He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition.
    CoRR, abs/1512.03385. http://arxiv.org/abs/1512.03385

    Parameters
    ----------
    input_shape : int
        Shape of the input in the format of batch_size x channels x time x pitch_range.
    num_channels : int
        Base number of channels. Each encoder block will double the number of channels from the previous
        encoder block, based on this variable.
    """
    def __init__(self, num_channels, input_shape):
        super().__init__()

        self.shapes = [(input_shape[2], input_shape[3]), (32, 44), (16, 22)]

        self.network = nn.Sequential(
            self.Encoder(input_shape[1], num_channels, 7, 1, 3),
            self.Encoder(num_channels, num_channels*2, 3, 2, 1),
            self.Encoder(num_channels*2, num_channels*4, 3, 2, 1),

            ResnetBlock(num_channels*4, num_channels*4),
            ResnetBlock(num_channels*4, num_channels*4),
            ResnetBlock(num_channels*4, num_channels*4),
            ResnetBlock(num_channels*4, num_channels*4),
            ResnetBlock(num_channels*4, num_channels*4),
            ResnetBlock(num_channels*4, num_channels*8),
            ResnetBlock(num_channels*8, num_channels*8),
            ResnetBlock(num_channels*8, num_channels*4),
            ResnetBlock(num_channels*4, num_channels*4),
            ResnetBlock(num_channels*4, num_channels*4),
            ResnetBlock(num_channels*4, num_channels*4),
            ResnetBlock(num_channels*4, num_channels*4),
        )

        self.upconv3 = torch.nn.ConvTranspose2d(num_channels*4, num_channels*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.upconv2 = torch.nn.ConvTranspose2d(num_channels*2, num_channels, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn_relu2 = self.Decoder(num_channels*2)
        self.bn_relu1 = self.Decoder(num_channels)

        self.out_conv = nn.Conv2d(64, input_shape[1], kernel_size=7, padding=3, stride=1, bias=False)

    def forward(self, x):
        out = self.network(x)

        out = self.upconv3(out, output_size=self.shapes[1])
        out = self.bn_relu2(out)

        out = self.upconv2(out, output_size=self.shapes[0])
        out = self.bn_relu1(out)

        out = self.out_conv(out)
        return torch.sigmoid(out)

    def Encoder(self, input_kernels, output_kernels, kernel_size=7, stride=1, padding=3):
        return nn.Sequential(
            nn.Conv2d(input_kernels, output_kernels, kernel_size, stride, padding, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(output_kernels),
            nn.ReLU(),
        )

    def Decoder(self, output_kernels):
        return nn.Sequential(
            nn.InstanceNorm2d(output_kernels),
            nn.ReLU(),
        )

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock, self).__init__()
        self.alter_size = True if in_channels != out_channels else False
        if self.alter_size:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn_1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn_2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        if self.alter_size:
            x_prev = self.conv1x1(x)

        out = self.conv1(x)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn_2(out)
        out = out + x_prev if self.alter_size else out + x
        return self.relu(out)
