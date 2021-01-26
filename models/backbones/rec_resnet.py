# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-26

import torch.nn as nn

from . import register_rec_backbone
from ..layers import Conv2dBNLayer, BottleneckBlock, BasicResBlock


@register_rec_backbone('ResNet')
class ResNet(nn.Module):
    def __init__(self, in_channels=3, layers=50, **kwargs):
        super(ResNet, self).__init__()

        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, f"supported layers are {supported_layers} but input layer in {layers}"

        # 网络的默认配置
        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv1_1 = Conv2dBNLayer(in_channels=in_channels,
                                     out_channels=32,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     act='relu',
                                     bias=False)
        self.conv1_2 = Conv2dBNLayer(in_channels=32,
                                     out_channels=32,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     bias=False,
                                     act='relu')
        self.conv1_3 = Conv2dBNLayer(in_channels=32,
                                     out_channels=64,
                                     padding=1,
                                     kernel_size=3,
                                     stride=1,
                                     bias=False,
                                     act='relu')
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block_list = nn.Sequential()
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    if i == 0 and block != 0:
                        # 每一块的第一个卷积，除第一块以外
                        stride = (2, 1)
                    else:
                        stride = (1, 1)

                    self.block_list.add_module(
                        name=f"bb_{block}_{i}",
                        module=BottleneckBlock(
                            in_channels=num_channels[block] if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=stride,
                            shortcut=shortcut,
                            if_first=block == i == 0))
                    shortcut = True
                self.out_channels = num_filters[block]
        else:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    stride = (2, 1) if i == 0 and block != 0 else (1, 1)

                    self.block_list.add_module(
                        name=f'bb_{block}_{i}',
                        module=BasicResBlock(
                            in_channels=num_channels[block] if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=stride,
                            shortcut=shortcut,
                            if_first=block == i == 0))
                    shortcut = True
                self.out_channels = num_filters[block]
        self.out_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        y = self.conv1_1(x)
        y = self.conv1_2(y)
        y = self.conv1_3(y)

        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        y = self.out_pool(y)
        return y
