# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2020/11/16

__all__ = ['build_backbone']


def build_backbone(config: dict, model_type: str):
    if model_type == 'rec':
        from .rec_mobile_v3 import MobileNetV3
        support_dict = ['MobileNetV3']
    else:
        raise NotImplemented

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'when model typs is {}, backbone only support {}'.format(model_type,
                                                                 support_dict))
    module_class = eval(module_name)(**config)
    return module_class
