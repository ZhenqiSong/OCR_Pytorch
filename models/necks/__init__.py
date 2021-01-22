# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-21

__all__ = ['build_neck']


def build_neck(config):
    from .rnn import SequenceEncoder
    support_dict = ['SequenceEncoder']

    module_name = config.pop('name')
    assert module_name in support_dict, f'neck only support {support_dict}'

    module_class = eval(module_name)(**config)
    return module_class
