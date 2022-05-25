"""All models defined in this file run some form of convolution to generalise data.

Avoid running models/classifiers here. Resort to running them in your main file by importing them instead
"""

from torch import nn
import numpy as np

import sys
sys.path.insert(1, '../')

class SimpleCNN(nn.Module):
    """This CNN structures itself by adding layers until the 'width' or the 'height' of the input reaches
    around 8. Therefore, there is no need to input how many layers. Dropout is optional, and
    will be added before the last linear layer (which there are three of).

    Attributes
    ----------
    k_size : int
        Contains size of kernel during convolution ((k_size, k_size) will be the shape of the kernel).
    padding: int
        Contains how much to pad during convolution ((channels, width + padding, height + padding) will be the
        shape of the padded input).
    stride: int
        Contains how large the stride is during convolution (a convolution will move the kernel 'stride' steps
        each time).
    maxpool_size: int
        Contains size of kernel during maxpool ((maxpool_size, maxpool_size) will be the shape of the kernel).
    use_dropout: bool
        Gets automatically assigned a value depending on the user input of dropout (please see attribute
        'dropout').

    """
    def __init__(self,
                num_channels,
                input_shape,
                dropout:float=0,
                channel_increase_factor:float=2.0,
                **kwargs):

        super(SimpleCNN, self).__init__()

        # kernel size, padding, maxpool kernel size
        self.k_size = 7
        self.padding = 3
        self.stride = 1
        self.maxpool_size = 2

        # list of layers for sequential model
        conv_modules = []

        # input layer, add 1 channel for bias channel and another 1 if it is now uneven
        num_ch = (num_channels)*channel_increase_factor
        conv_modules.append(self._conv_layer_set(num_channels, int(num_ch), input_layer=True))

        # keep track of dimensions of tensor
        # shape after every convolution + maxpool operation:
            # width = 1+(w-k+2p)/s*x       w-width, k-kernel size, p-padding, s-stride, x-maxpool kernel size
            # height = 1+(h-k+2p)/s*x     h-height, k-kernel size, p-padding, s-stride, x-maxpool kernel size

        shape = (int((1+(input_shape[0]-self.k_size+2*self.padding)/(self.stride))/self.maxpool_size),
                 int((1+(input_shape[1]-self.k_size+2*self.padding)/(self.stride))/self.maxpool_size))

        # keep adding layers until width or height is close to 8
        # also increase num_ch for each iteration
        while min(shape) > 2:
            conv_modules.append(self._conv_layer_set(int(num_ch), int(num_ch*channel_increase_factor)))
            num_ch *= channel_increase_factor
            shape = (int((1+(shape[0]-self.k_size+2*self.padding)/(self.stride))/self.maxpool_size),
                     int((1+(shape[1]-self.k_size+2*self.padding)/(self.stride))/self.maxpool_size))

        # add linear layers, using calculated shapes
        linear_modules = []
        # cast to int as this may generate a float value
        inp_shape = int(shape[0] * shape[1] * num_ch)

        scale = inp_shape//2
        linear1 = nn.Linear(inp_shape, int(inp_shape//(scale*0.25)))
        linear2 = nn.Linear(int(inp_shape//(scale*0.25)), int(inp_shape//(scale*0.75)))
        linear3 = nn.Linear(int(inp_shape//(scale*0.75)), 1)

        linear_modules.extend([linear1, linear2])
        if 0 < dropout < 1:
            drop = nn.Dropout(dropout)
            linear_modules.append(drop)

        linear_modules.append(linear3)

        # Set the output activation
        sigmoid = nn.Sigmoid()
        linear_modules.append(sigmoid)

        # put all layers in sequential model
        self.convs = len(conv_modules)
        self._conv = nn.Sequential(*conv_modules)
        self._linear = nn.Sequential(*linear_modules)

    def _conv_layer_set(self, in_c, out_c, input_layer: bool=False):
        # Set the convolutional layers, batch normalization for all layers except input
        conv_modules = []

        conv_modules.append(nn.Conv2d(in_c, out_c, kernel_size=self.k_size, padding=self.padding, bias=False))
        if not input_layer: conv_modules.append(nn.BatchNorm2d(out_c))
        conv_modules.append(nn.LeakyReLU(0.2))
        conv_modules.append(nn.MaxPool2d(self.maxpool_size))

        conv_layer = nn.Sequential(*conv_modules)
        return conv_layer


    def forward(self, x):
        out = self._conv(x)
        out = out.view(out.size(0), -1) # Flatten (batchsize, image size)
        out = self._linear(out)
        return out
