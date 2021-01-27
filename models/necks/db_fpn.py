# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-26

import torch
import torch.nn as nn
from torch.nn import functional as F
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
        c2, c3, c4, c5 = x
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in4_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.upsample(in5, scale_factor=2, mode='nearest')
        out3 = in3 + F.upsample(out4, scale_factor=2, mode='nearest')
        out2 = in2 + F.upsample(out3, scale_factor=2, mode='nearest')

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)

        p5 = F.upsample(p5, scale_factor=8, mode='nearest')
        p4 = F.upsample(p4, scale_factor=4, mode='nearest')
        p3 = F.upsample(p3, scale_factor=2, mode='nearest')

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse
