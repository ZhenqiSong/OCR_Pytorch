# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2020/12/3

import torch
import torch.nn as nn




class ResNetCls(nn.Module):
    def __init__(self, class_nums):
        super(ResNetCls, self).__init__()

        self.backbone = ResNet()