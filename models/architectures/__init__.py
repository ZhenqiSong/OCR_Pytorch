# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-20

import copy

__all__ = ['build_model']


def build_model(config):
    from .base_model import BaseModel

    config = copy.deepcopy(config)
    module_class = BaseModel(config)
    return module_class