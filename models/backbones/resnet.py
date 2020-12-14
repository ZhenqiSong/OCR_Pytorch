# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2020/12/3


import torch
import torch.nn as nn
from torchvision.models import ResNet

from models.layers.custom_module import Conv2dBNLayer

__all__ = ['resnet18']


class BasicBlock(nn.Module):
    """层数小于50时的基础块结构"""
    def __init__(self, in_filters, out_filters, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dBNLayer(in_filters, out_filters, 3, act='relu', stride=stride, padding=1)
        self.conv2 = Conv2dBNLayer(out_filters, out_filters, 3, act='', stride=1, padding=1)



class ResNet(nn.Module):
    def __init__(self, block, layers):
        pass


def resnet18(*args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)