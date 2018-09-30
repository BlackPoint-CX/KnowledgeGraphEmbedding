#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : ConfigBuilder.py
__create_time__ : 2018/09/27
"""


class ConfigBuilder(object):

    def __init__(self):
        self.ent_total_num = None  # Number of all entities
        self.rel_total_num = None  # Number of all relations
        self.batch_size = None  # Batch size
        self.batch_seq_size = None # TODO
        self.optimizer = None  # Choice of optimizer
        self.lr = None
        self.lr_decay = None


        pass
