# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-25

import paddle
from torch import nn
import torch
import numpy as np
from paddle import ParamAttr


def do_paddle(nx, state_dict):
    x = paddle.to_tensor(nx, dtype='float32')
    rnn = paddle.nn.LSTM(288, 48, 2, direction='bidirectional')
    for k,v in rnn.state_dict().items():
        print(k, v.shape)
    for k, v in rnn.state_dict().items():
        if k in state_dict:
            v.set_value(state_dict[k])
    y, (h, c) = rnn(x)
    return y.numpy()


def do_torch(nx, state_dict):
    rnn = nn.LSTM(288, 48, 2, bidirectional=True, batch_first=True)
    for k, v in rnn.named_parameters():
        with torch.no_grad():
            v.copy_(torch.from_numpy(state_dict[k]))
    x = torch.from_numpy(nx)
    y, _ = rnn(x)
    return y.detach().numpy()


if __name__ == '__main__':
    x = np.random.randn(1, 2, 288).astype(np.float32)
    in_dim = 288
    hidden_size = 48
    state_dict = {'weight_ih_l0': np.random.randn(hidden_size * 4, in_dim).astype(np.float32),
                  'weight_hh_l0': np.random.randn(hidden_size * 4, hidden_size).astype(np.float32),
                  'bias_ih_l0': np.random.randn(hidden_size*4).astype(np.float32),
                  'bias_hh_l0': np.random.randn(hidden_size*4).astype(np.float32),
                  'weight_ih_l0_reverse': np.random.randn(hidden_size * 4, in_dim).astype(np.float32),
                  'weight_hh_l0_reverse': np.random.randn(hidden_size * 4, hidden_size).astype(np.float32),
                  'bias_ih_l0_reverse': np.random.randn(hidden_size * 4).astype(np.float32),
                  'bias_hh_l0_reverse': np.random.randn(hidden_size * 4).astype(np.float32),

                  'weight_ih_l1': np.random.randn(hidden_size*4, hidden_size*2).astype(np.float32),
                  'weight_hh_l1': np.random.randn(hidden_size*4, hidden_size).astype(np.float32),
                  'bias_ih_l1': np.random.randn(hidden_size*4).astype(np.float32),
                  'bias_hh_l1': np.random.randn(hidden_size*4).astype(np.float32),
                  'weight_ih_l1_reverse': np.random.randn(hidden_size*4, hidden_size*2).astype(np.float32),
                  'weight_hh_l1_reverse': np.random.randn(hidden_size*4, hidden_size).astype(np.float32),
                  'bias_ih_l1_reverse': np.random.randn(hidden_size*4).astype(np.float32),
                  'bias_hh_l1_reverse': np.random.randn(hidden_size*4).astype(np.float32),
                  }
    # print(state_dict)

    res_paddle = do_paddle(x, state_dict)
    res_torch = do_torch(x, state_dict)
    print(np.sum(res_paddle - res_torch))
