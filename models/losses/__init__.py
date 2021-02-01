# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-02-01

import os
import importlib
import copy

LOSS_REGISTRY = {}


def build_loss(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if module_name not in LOSS_REGISTRY:
        raise ModuleNotFoundError('loss only support {}'.format(list(LOSS_REGISTRY.keys())))

    model_class = LOSS_REGISTRY[module_name](**config)
    return model_class


def register_loss(name):
    def register_model_cls(cls):
        if name in LOSS_REGISTRY:
            raise ValueError("Cannot register duplicate loss ({})".format(name))
        LOSS_REGISTRY[name] = cls
        return cls
    return register_model_cls


for file in os.listdir(os.path.dirname(__file__)):
    if not file.startswith('.') and not file.startswith('_') and file.endswith('.py'):
        model_name = file[: file.find(".py")] if file.endswith('.py') else file
        module = importlib.import_module("models.losses." + model_name)