# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-02-01

import copy
import torch
from . import lr_rate
from torch.optim import Adam, lr_scheduler


def build_lr_scheduler(config: dict, epochs, step_each_epoch):
    config.update({'epochs': epochs, 'step_each_epoch': step_each_epoch})
    if 'name' in config:
        lr_name = config.pop('name')
        lr = getattr(lr_rate, lr_name)(**config)()
    else:
        lr = config['learning_rate']
    return lr


def build_optimizer(config, epochs, step_each_epoch, parameters):
    from . import regularizer, optimizer
    config = copy.deepcopy(config)

    optim_name = config.pop('name')
    optim_config = config.pop('optim_parms')
    optimizer = eval(optim_name)(parameters, **optim_config)

    lr_config = config.pop('lr')
    lr_name = lr_config['name']
    if lr_name == 'Cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs * step_each_epoch,
            last_epoch=lr_config.get('last_epoch', -1)
        )

    return optimizer, scheduler

