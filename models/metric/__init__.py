# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-02-01


__all__ = ['build_metric']
import copy
from .rec_metric import RecMetric


def build_metric(config):
    support_dict = ['DetMetric', 'RecMetric', 'ClsMetric']
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict

    module_class = eval(module_name)(**config)
    return module_class