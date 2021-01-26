# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2020/11/16


import os
import importlib
from torch import nn as nn

__all__ = ['build_backbone']

REC_BACKBONE_REGISTRY = {}
DET_BACKBONE_REGISTRY = {}


def build_backbone(config: dict, model_type: str):
    module_name = config.pop('name')
    model_registry = {}
    if model_type == 'rec':
        model_registry = REC_BACKBONE_REGISTRY
    elif model_type == 'det':
        model_registry = DET_BACKBONE_REGISTRY
    assert module_name in model_registry, Exception(
        'when model type is {}, backbone only support {}'.format(model_type,
                                                                 list(model_registry.keys())))
    module_class = model_registry[module_name](**config)
    return module_class


def register_det_backbone(name):
    def register_model_cls(cls):
        if name in DET_BACKBONE_REGISTRY:
            raise ValueError("Cannot register duplicate det backbone ({})".format(name))
        if not issubclass(cls, nn.Module):
            raise ValueError("({} : {}) is not a valid Module".format(name, cls.__name__))
        DET_BACKBONE_REGISTRY[name] = cls
        return cls
    return register_model_cls


def register_rec_backbone(name):
    def register_model_cls(cls):
        if name in REC_BACKBONE_REGISTRY:
            raise ValueError("Cannot register duplicate rec backbone ({})".format(name))
        if not issubclass(cls, nn.Module):
            raise ValueError("({} : {}) is not a valid Module".format(name, cls.__name__))
        REC_BACKBONE_REGISTRY[name] = cls
        return cls
    return register_model_cls


model_dir = os.path.dirname(__file__)
for file in os.listdir(model_dir):
    if not file.startswith("_") and not file.startswith(".") and file.endswith('.py'):
        model_name = file[: file.find(".py")] if file.endswith('.py') else file
        module = importlib.import_module("models.backbones." + model_name)