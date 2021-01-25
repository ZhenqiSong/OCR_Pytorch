# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-21

import os
import importlib
from torch import nn as nn

__all__ = ['build_head']

HEAD_REGISTRY = {}


def build_head(config):
    module_name = config.pop('name')
    assert module_name in HEAD_REGISTRY, Exception('head only support {}'.format(
        list(HEAD_REGISTRY.keys())))
    module_class = HEAD_REGISTRY[module_name](**config)
    return module_class


def register_head(name):
    def register_model_cls(cls):
        if name in HEAD_REGISTRY:
            raise ValueError("Cannot register duplicate head module ({})".format(name))
        if not issubclass(cls, nn.Module):
            raise ValueError("({} : {}) is not a valid Module".format(name, cls.__name__))
        HEAD_REGISTRY[name] = cls
        return cls

    return register_model_cls


for file in os.listdir(os.path.dirname(__file__)):
    if not file.startswith("_") and not file.startswith(".") and file.endswith('.py'):
        model_name = file[: file.find('.py')] if file.endswith('.py') else file
        importlib.import_module('models.heads.' + model_name)
