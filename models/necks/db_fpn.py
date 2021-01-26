# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-26

import torch.nn as nn
from . import register_neck


@register_neck('DBFPN')
class DBFPN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DBFPN, self).__init__()

        self.out_channels = out_channels

        self.in2_conv = nn.Conv2d(in_channels=in_channels[0],
                                  out_channels=self.out_channels,
                                  kernel_size=1,
                                  bias=False)

        self.in3_conv = nn.Conv2d(in_channels=in_channels[1],
                                  out_channels=self.out_channels,
                                  kernel_size=1,
                                  bias=False)

        self.in4_conv = nn.Conv2d(in_channels=in_channels[2],
                                  out_channels=self.out_channels,
                                  kernel_size=1,
                                  bias=False)

        self.in5_conv = nn.Conv2d(in_channels=in_channels[3],
                                  out_channels=self.out_channels,
                                  kernel_size=1,
                                  bias=False)

        self.p5_conv = nn.Conv2d(in_channels=self.out_channels,
                                 out_channels=self.out_channels//4,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False)
        self.p4_conv = nn.Conv2d(in_channels=self.out_channels,
                                 out_channels=self.out_channels//4,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False)
        self.p3_conv = nn.Conv2d(in_channels=self.out_channels,
                                 out_channels=self.out_channels//4,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False)

        self.p2_conv = nn.Conv2d(in_channels=self.out_channels,
                                 out_channels=self.out_channels//4,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False)

    def forward(self, x):
        pass