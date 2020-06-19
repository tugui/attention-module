import torch
import math
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, activation=None, bn=True, bias=False):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True) if bn else None
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.leakyReLU()
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class SpatialGate(nn.Module):
    def __init__(self, in_channel, kernel_size=3, block_num=0):
        super(SpatialGate, self).__init__()
        if block_num > 0:
            self.layer = self._make_layer(in_channel, block_num)
        else:
            self.layer = None
        self.spatial = BasicBlock(in_channel, 1, kernel_size, padding=(kernel_size-1) // 2)

    def forward(self, x):
        x_out = self.spatial(self.layer(x) if self.layer is not None else x)
        scale = torch.sigmoid(x_out)
        return x * scale

    def _make_layer(self, in_channel, block_num):
        layers = []
        for i in range(block_num):
            layers.append(BasicBlock(in_channel, in_channel, kernel_size, padding=(kernel_size-1) // 2))
        return nn.Sequential(*layers)

class SpatialGate2(nn.Module):
    def __init__(self, in_channel, kernel_size=3, block_num=0):
        super(SpatialGate2, self).__init__()
        if block_num > 0:
            self.layer = self._make_layer(in_channel, block_num)
        else:
            self.layer = None
        self.spatial = BasicBlock(in_channel, 1, kernel_size, padding=(kernel_size-1) // 2)

    def forward(self, x):
        x_out = self.spatial(self.layer(x) if self.layer is not None else x)
        scale = torch.tanh(x_out)
        return x * scale

    def _make_layer(self, in_channel, block_num):
        layers = []
        for i in range(block_num):
            layers.append(BasicBlock(in_channel, in_channel, kernel_size, padding=(kernel_size-1) // 2))
        return nn.Sequential(*layers)

class SAM(nn.Module):
    def __init__(self, in_channel, kernel_size, block_num):
        super(SAM, self).__init__()
        self.SpatialGate = SpatialGate(in_channel, kernel_size, block_num)

    def forward(self, x):
        return self.SpatialGate(x)

class SAM2(nn.Module):
    def __init__(self, in_channel, kernel_size, block_num):
        super(SAM2, self).__init__()
        self.SpatialGate = SpatialGate2(in_channel, kernel_size, block_num)

    def forward(self, x):
        return self.SpatialGate2(x)
