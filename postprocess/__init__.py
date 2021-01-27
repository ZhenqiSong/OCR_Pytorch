# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-22

import copy
import os
import importlib

__all__ = ['build_post_process']

POST_REGISTRY = {}


def build_post_process(config, global_config=None):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if global_config is not None:
        config.update(global_config)
    assert module_name in POST_REGISTRY, Exception(
        'post process only support {}'.format(list(POST_REGISTRY.keys())))
    module_class = POST_REGISTRY[module_name](**config)
    return module_class


def register_post(name):
    def register_model_cls(cls):
        if name in POST_REGISTRY:
            raise ValueError("Cannot register duplicate postprocess ({})".format(name))
        POST_REGISTRY[name] = cls
        return cls
    return register_model_cls


for file in os.listdir(os.path.dirname(__file__)):
    if not file.startswith('.') and not file.startswith('_') and file.endswith('.py'):
        model_name = file[: file.find('.py')] if file.endswith('.py') else file
        importlib.import_module('postprocess.' + model_name)
