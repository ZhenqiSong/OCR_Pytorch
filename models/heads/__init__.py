# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-21


__all__ = ['build_head']


def build_head(config):
    from .rec_ctc_head import CTCHead

    support_dict = ['CTCHead']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('head only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class