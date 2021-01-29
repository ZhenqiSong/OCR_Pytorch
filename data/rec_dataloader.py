# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-28

import os
import numpy as np
import random
from torch.utils.data import Dataset
from .aug import create_transformers


class RecDataset(Dataset):
    def __init__(self, config, mode, logger):
        super(RecDataset, self).__init__()
        self.logger = logger

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", [1.0])
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(ratio_list) == data_source_num, "The length of ratio_list should be the same as the file list"
        self.data_dir = dataset_config['data_dir']
        if not os.path.exists(self.data_dir):
            raise FileExistsError("图像路径: {} 不存在!".format(self.data_dir))

        self.do_shuffle = loader_config['shuffle']

        self.data_lines = self.get_data_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        self.transforms = create_transformers(dataset_config['transforms'], global_config)

    def get_data_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                lines = random.sample(lines, round(len(lines)*ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            # 读取标签中的一行数据
            data_line = data_line.decode('utf-8')
            substr = data_line.strip('\n').split(self.delimiter)
            file_name = substr[0]
            label = substr[1]

            # 读取图像数据
            img_path = os.path.join(self.data_dir, file_name)
            data = {'image': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} 不存在！".format(img_path))
            outs = self.transforms(data)

        except Exception as e:
            self.logger.error('读取数据： "{}", 发生错误： {} '.format(data_line.rstrip('\n'), e))
            outs = None
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)