# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-26

import paddle
import torch

paddle_dict = paddle.load(r'D:\SongZhenqi\model\OCR\paddle\2.0\ch_ppocr_server_v2.0_rec_pre\best_accuracy.pdparams')
torch_dict = torch.load('../resnet_crnn_ctc.pth')

a = '123'

for k, v in torch_dict.items():

    p_k = k
    if k.startswith('backbone'):
        p_k = k.replace('.block_list', '')

        if k.find('.conv.'):
            p_k = p_k.replace('.conv.', '._conv.')

        if k.find('.bn.'):
            p_k = p_k.replace('.bn.', '._batch_norm.')
            if p_k.endswith('running_mean'):
                p_k = p_k.replace('running_mean', '_mean')
            elif p_k.endswith('running_var'):
                p_k = p_k.replace('running_var', '_variance')
            elif p_k.endswith('num_batches_tracked'):
                continue

    data = paddle_dict[p_k]
    if p_k.endswith('head.fc.weight'):
        data = data.transpose(-1, -2)

    assert data.shape == v.numpy().shape

    torch_dict[k] = torch.from_numpy(data)

torch.save(torch_dict, '../resnet_crnn_ctc.pth')



