# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-02-01

import torch

class L1Decay(object):
    def __init__(self, factor=0.0):
        super(L1Decay, self).__init__()
        self.regularization_coeff = factor

    def __call__(self):
        # reg = torch.optim.
        pass