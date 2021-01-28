# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-26

import os
from argparse import ArgumentParser
import torch
import numpy as np
import json
import cv2 as cv

from utils import get_logger, get_config, get_img_list
from models.architectures import build_model
from postprocess import build_post_process
from data import create_transformers


def draw_det_res(dt_boxes, save_path, img, img_name):
    if len(dt_boxes) > 0:
        src_im = img
        for box in dt_boxes:
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        save_det_path = os.path.join(save_path, 'det_results')
        if not os.path.exists(save_det_path):
            os.makedirs(save_det_path)
        save_name = os.path.join(save_det_path, os.path.basename(img_name))
        cv.imwrite(save_name, src_im)
        logger.info("结果图像保存在：{}".format(save_name))


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--cfg', '-c', help='Config file', required=True)

    return parser.parse_args()


def main():
    global_config = config['Global']

    device = torch.device('cpu')
    if global_config['use_gpu'] and torch.cuda.is_available():
        device = torch.device('cuda')
    logger.info('使用设备：{}'.format(device))

    logger.info('模型信息：{}'.format(config['Architecture']))
    model = build_model(config['Architecture'])
    model.to(device)

    logger.info('加载预训练模型：{}'.format(global_config['pretrained_model']))
    state_dict = torch.load(global_config['pretrained_model'])
    model.load_state_dict(state_dict)

    post_process_class = build_post_process(config['PostProcess'])

    ops = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == "KeepKeys":
            op[op_name]['keep_keys'] = ['image', 'shape']
        ops.append(op)
    transforms = create_transformers(ops)

    save_res_path = global_config['save_res_path']
    save_dir = os.path.dirname(save_res_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    with open(save_res_path, 'wb') as fout:
        for file in get_img_list(global_config['infer_img']):
            logger.info(f"测试图像：{file}")
            data = {'image': file}
            batch = transforms(data)

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = torch.from_numpy(images).to(device)
            preds = model(images)
            post_result = post_process_class(preds, shape_list)
            boxes = post_result[0]['points']

            dt_boxes_json = []
            for box in boxes:
                tmp_json = {"transcription": ""}
                tmp_json['points'] = box.tolist()
                dt_boxes_json.append(tmp_json)
            otstr = file + "\t" + json.dumps(dt_boxes_json) + '\n'
            fout.write(otstr.encode())
            src_img = cv.imread(file)
            draw_det_res(boxes, save_dir, src_img, file)
        logger.info("结果已保存！")


if __name__ == '__main__':
    args = get_args()
    config = get_config(args.cfg)
    logger = get_logger()
    logger.info("当前配置文件为：{}".format(args.cfg))
    main()
