# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-20

from collections import OrderedDict
import itertools

import torch
from typing import Union, Dict
from torch import nn
import numpy as np

from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head


class BaseModel(nn.Module):
    def __init__(self, config):
        """
        OCR model
        :param config:
        """
        super(BaseModel, self).__init__()

        in_channels = config.get('in_channels', 3)
        model_type = config['model_type']

        config["Backbone"]['in_channels'] = in_channels
        self.backbone = build_backbone(config["Backbone"], model_type)
        in_channels = self.backbone.out_channels

        # Neck
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels
        #
        # # Head
        # config["Head"]['in_channels'] = in_channels
        # self.head = build_head(config["Head"])

    def forward(self, x):
        x = self.backbone(x)

        if self.use_neck:
            x = self.neck(x)

        x = self.head(x)
        return x

    def load_paddle_state_dict(self, state_dict: Union[Dict[str, np.array], str], strict: bool = True):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        metadata = OrderedDict()
        for k, v in state_dict.items():
            key = '.'.join(k.split('.')[:-1])
            metadata[key] = metadata.get(key, 0) + 1

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            _load_from_paddle_state_dict(module, prefix, local_metadata, True)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        def _load_from_paddle_state_dict(module, prefix, local_metadata, strict):
            persistent_buffers = {k: v for k, v in module._buffers.items() if
                                  k not in module._non_persistent_buffers_set}
            local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
            local_state = {k: v for k, v in local_name_params if v is not None}

            for name, param in local_state.items():
                paddle_name = self._check_param_name(name)
                key = prefix + name
                paddle_key = prefix + paddle_name
                if paddle_key in state_dict:
                    input_param = torch.from_numpy(state_dict[paddle_key])
                    input_param = self._check_param_size(module, input_param)

                    if len(param.shape) == 0 and len(input_param.shape) == 1:
                        input_param = input_param[0]
                    if input_param.shape != param.shape:
                        error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                          'the shape in current model is {}.'
                                          .format(key, input_param.shape, param.shape))
                        continue
                    try:
                        with torch.no_grad():
                            param.copy_(input_param)
                    except Exception as ex:
                        error_msgs.append('While copying the parameter named "{}", '
                                          'whose dimensions in the model are {} and '
                                          'whose dimensions in the checkpoint are {}, '
                                          'an exception occurred : {}.'
                                          .format(key, param.size(), input_param.size(), ex.args))
                elif strict:
                    missing_keys.append(key)

            if strict:
                for key in state_dict.keys():
                    if key.startswith(prefix):
                        input_name = key[len(prefix):]
                        input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                        input_name = self._check_param_name(input_name)
                        if input_name not in module._modules and input_name not in local_state:
                            unexpected_keys.append(key)

        load(self)
        load = None

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        for error_msg in error_msgs:
            print(error_msg)

    def _check_param_name(self, key):
        matched_keys = {'running_mean': '_mean',
                        '_mean': 'running_mean',
                        'running_var': '_variance',
                        '_variance': 'running_var'}

        return matched_keys.get(key, key)

    def _check_param_size(self, module, param: torch.Tensor):
        if isinstance(module, nn.Linear) and len(param.shape) > 1:
            param = param.transpose(-1, -2)
        return param
