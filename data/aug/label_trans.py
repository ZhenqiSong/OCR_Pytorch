# -*- coding: utf-8 -*-
# __author__:Song Zhenqi
# 2021-01-29

import numpy as np


class BaseRecLabelEncode(object):
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False):

        support_char_type=['ch', 'en']

        assert character_type in support_char_type, "{} is not a support char in {}".format(
            use_space_char, support_char_type)

        self.max_length = max_text_length
        if character_type == 'en':
            self.char_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_char = list(self.char_str)
        elif character_type == 'ch':
            self.char_str = ""
            assert character_dict_path is not None, "必须提供字典文件"

            with open(character_dict_path, "rb") as fi:
                lines = fi.readlines()

            self.char_str = ''.join([line.decode('utf-8').strip('\n').strip('\r\n') for line in lines])
            if use_space_char:
                self.char_str += " "
            dict_char = list(self.char_str)

        self.char_type = character_type
        dict_char = self.add_special_char(dict_char)
        self.dict = {char: i for i, char in enumerate(dict_char)}
        self.chars = dict_char

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        if len(text) == 0 and len(text) > self.max_length:
            return None

        if self.char_type == 'en':
            text = text.lower()

        text_list = [self.dict[char] for char in text if char in self.dict]
        return text_list if len(text_list) > 0 else None


class CTCLabelEncode(BaseRecLabelEncode):
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelEncode, self).__init__(max_text_length, character_dict_path,
                                             character_type, use_space_char)

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None

        data['length'] = np.array(len(text))
        text = text + [0] * (self.max_length - len(text))
        data['label'] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character
