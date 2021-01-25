# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-21

import os
import importlib
from torch import nn

__all__ = ['build_neck']

NECK_REGISTRY = {}


def build_neck(config):
    module_name = config.pop('name')
    assert module_name in NECK_REGISTRY, f'neck only support {list(NECK_REGISTRY.keys())}'

    module_class = NECK_REGISTRY[module_name](**config)
    return module_class


def register_neck(name):
    def register_model_cls(cls):
        if name in NECK_REGISTRY:
            raise ValueError("Cannot register duplicate neck module ({})".format(name))
        if not issubclass(cls, nn.Module):
            raise ValueError("({} : {}) is not a valid module".format(name, cls.__name__))
        NECK_REGISTRY[name] = cls
        return cls
    return register_model_cls


for file in os.listdir(os.path.dirname(__file__)):
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        importlib.import_module('models.necks.'+model_name)