# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-28

from torch.utils.data import Dataset


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

    def forward(self, x):
        pass