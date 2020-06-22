import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from .cbam import *
from .bam import *
from .lsam import LSAM
from .lsam2 import LSAM2
from .se import *
from .sam import *

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, att_type=None, position=0, block_num=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if att_type == 'CBAM':
            self.cbam = CBAM(planes, 16)
        else:
            self.cbam = None

        if att_type == 'SE':
            self.se = SE(planes, 16)
        else:
            self.se = None

        # if att_type == 'LSAM':
        #     self.lsam = LSAM(planes, 3, block_num, planes)
        # else:
        #     self.lsam = None

        # if att_type == 'SAM':
        #     self.sam = SAM(planes, 3, block_num)
        # else:
        #     self.sam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.position == 2:
            out = self.attention(out)

        out += residual
        out = self.relu(out)

        if self.position == 1:
            out = self.attention(out)

        return out

    def attention(self, x):
        if self.cbam is not None:
            out = self.cbam(x)
        elif self.se is not None:
            out = self.se(x)
        # elif self.sam is not None:
        #     out = self.sam(x)
        # elif self.lsam is not None:
        #     out = self.lsam(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, att_type=None, position=0, block_num=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.position = position
        self.block_num = block_num

        if att_type == 'CBAM':
            self.cbam = CBAM(planes * 4, 16)
        else:
            self.cbam = None

        if att_type == 'SE':
            self.se = SE(planes * 4, 16)
        else:
            self.se = None

        # if att_type == 'LSAM':
        #     self.lsam = LSAM(planes * 4, 3, block_num, planes)
        # else:
        #     self.lsam = None

        # if att_type == 'SAM':
        #     self.sam = SAM(planes * 4, 3, block_num)
        # else:
        #     self.sam = None

        

    def attention(self, x):
        if self.cbam is not None:
            out = self.cbam(x)
        elif self.se is not None:
            out = self.se(x)
        # elif self.sam is not None:
        #     out = self.sam(x)
        # elif self.lsam is not None:
        #     out = self.lsam(x)
        return x

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.position == 2:
            out = self.attention(out)

        out += residual
        out = self.relu(out)

        if self.position == 1:
            out = self.attention(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, network_type, num_classes, att_type=None, position=0, block_num=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR 
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type == 'BAM':
            self.bam1 = BAM(64*block.expansion)
            self.bam2 = BAM(128*block.expansion)
            self.bam3 = BAM(256*block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        if att_type == 'COM':
            self.com1 = block(64*block.expansion, 64, att_type = None, position = 0, block_num = 0)
            self.com2 = block(128*block.expansion, 128, att_type = None, position = 0, block_num = 0)
            self.com3 = block(256*block.expansion, 256, att_type = None, position = 0, block_num = 0)
        else:
            self.com1, self.com2, self.com3 = None, None, None

        if att_type == 'SAM':
            self.sam1 = SAM(64*block.expansion, 3, block_num)
            self.sam2 = SAM(128*block.expansion, 3, block_num)
            self.sam3 = SAM(256*block.expansion, 3, block_num)
        else:
            self.sam1, self.sam2, self.sam3 = None, None, None

        if att_type == 'LSAM':
            self.lsam1 = LSAM(64*block.expansion, 3, block_num, 64*block.expansion)
            self.lsam2 = LSAM(128*block.expansion, 3, block_num, 128*block.expansion)
            self.lsam3 = LSAM(256*block.expansion, 3, block_num, 256*block.expansion)
        elif att_type == 'LSAM2':
            self.lsam1 = LSAM2(64*block.expansion, 3, block_num, 64*block.expansion)
            self.lsam2 = LSAM2(128*block.expansion, 3, block_num, 128*block.expansion)
            self.lsam3 = LSAM2(256*block.expansion, 3, block_num, 256*block.expansion)
        else:
            self.lsam1, self.lsam2, self.lsam3 = None, None, None

        self.layer1 = self._make_layer(block, 64,  layers[0], 1, att_type, position, block_num)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, att_type, position, block_num)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, att_type, position, block_num)
        self.layer4 = self._make_layer(block, 512, layers[3], 2, att_type, position, block_num)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None, position=0, block_num=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, att_type=att_type, position=position, block_num=block_num))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, att_type=att_type, position=position, block_num=block_num))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)
        elif not self.com1 is None:
            x = self.com1(x)
        elif not self.sam1 is None:
            x = self.sam1(x)
        elif not self.lsam1 is None:
            x = self.lsam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)
        elif not self.com2 is None:
            x = self.com2(x)
        elif not self.sam2 is None:
            x = self.sam2(x)
        elif not self.lsam2 is None:
            x = self.lsam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)
        elif not self.com3 is None:
            x = self.com3(x)
        elif not self.sam3 is None:
            x = self.sam3(x)
        elif not self.lsam3 is None:
            x = self.lsam3(x)

        x = self.layer4(x)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResidualNet(network_type, depth, num_classes, att_type, position, block_num):

    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type, position, block_num)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type, position, block_num)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type, position, block_num)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type, position, block_num)

    return model
