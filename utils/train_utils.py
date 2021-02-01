# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-02-01

import torch
from tqdm import tqdm
import time


def eval(model, dataloader, post_process, eval_class, device):
    model.eval()
    with torch.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(total=len(dataloader), desc='eval model:')

        for idx, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            images = batch[0]
            start = time.time()
            preds = model(images)

            batch = [item.detach().cpu().numpy() for item in batch]

            post_result = post_process(preds, batch[1])
            total_time += time.time() - start

            eval_class(post_result, batch)
            pbar.update(1)
            total_frame += len(images)

        metric = eval_class.get_metric()

    pbar.close()
    model.train()
    metric['fps'] = total_frame / total_time
    return metric