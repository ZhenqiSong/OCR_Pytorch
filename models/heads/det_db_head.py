# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-27

from . import register_head

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, in_channels):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels//4,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.conv_bn1 = nn.BatchNorm2d(num_features=in_channels//4)

        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels//4,
                                        out_channels=in_channels//4,
                                        kernel_size=2,
                                        stride=2)
        self.conv_bn2 = nn.BatchNorm2d(num_features=in_channels//4)

        self.conv3 = nn.ConvTranspose2d(in_channels=in_channels//4,
                                        out_channels=1,
                                        kernel_size=2,
                                        stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv_bn1(x), inplace=True)
        x = self.conv2(x)
        x = F.relu(self.conv_bn2(x), inplace=True)
        x = self.conv3(x)
        x = F.sigmoid(x)
        return x


@register_head('DBHead')
class DBHead(nn.Module):
    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k

        self.binarize = Head(in_channels)
        self.thresh = Head(in_channels)

    def forward(self, x):
        shrink_maps = self.binarize(x)
        if not self.training:
            return {'maps': shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat([shrink_maps, threshold_maps, binary_maps], dim=1)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x-y)))
