#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : trans_e_dataset.py
__create_time__ : 2018/09/30
"""
import random
from copy import deepcopy
import tensorflow as tf
from tensorflow.python.data import TextLineDataset
from tensorflow.python.data.ops.dataset_ops import Dataset


class DataManagerE(object):
    def __init__(self, file, config):
        self.config = config
        self.data_file = file
        self.data = []
        self.data_with_corrupted = []
        self.build_data()

    def build_data(self):
        """ Read file and push into data list """
        with open(self.data_file, 'r') as r_file:
            for line in r_file:
                h, r, t = line.strip().split('\t')
                self.data.append((h, r, t))

    def build_corrupt(self, h, r, t, mode, size):
        """
        According to mode ( 'h' or 't') to build corrupted triplet by replacing with random selected entity
        :return : list of (tuple((h,r,t),(h',r,t)) or tuple((h,r,t),(h,r,t')))
        """
        corrupt_list = []
        id_list = random.sample(range(self.config.ent_total_num), size)
        if mode == 'h':  # Remove Head Entity
            while h in id_list:
                id_list.remove(h)
                id_list.append(random.choice(range(self.config.ent_total_num)))
            for id in id_list:
                corrupt_list.append(((h, r, t), (id, r, t)))
        elif mode == 't':  # Remove Tail Entity
            while t in id_list:
                id_list.remove(h)
                id_list.append(random.choice(range(self.config.ent_total_num)))
            for id in id_list:
                corrupt_list.append(((h, r, t), (h, r, id)))
        assert len(corrupt_list) == size
        return corrupt_list

    def combine_with_corrupt(self, ori_triplet):
        """
        Sample #corrupt_sample_num from #corrupt_sample_size corrupted_triplet and combine them with ori_triplet
        :param ori_triplet: (h,r,t) record
        :return: #corrput_sample_num of (ori_triplet, corrupted_triplet)
        """
        h, r, t = ori_triplet
        replace_h_num = self.config.corrupt_sample_size // 2  # Num of (h', r, t)
        replace_t_num = self.config.corrupt_sample_size - replace_h_num  # Num of (h, r, t')
        replace_h_list = self.build_corrupt(h, r, t, 'h', replace_h_num)
        replace_t_list = self.build_corrupt(h, r, t, 't', replace_t_num)
        corrupt_list = []
        corrupt_list.extend(replace_h_list)
        corrupt_list.extend(replace_t_list)
        del replace_h_list
        del replace_t_list
        assert len(corrupt_list) == self.config.corrupt_sample_size
        return random.choices(corrupt_list, k=self.config.corrupt_sample_num)

    def __iter__(self):
        if self.config.shuffle:
            data = deepcopy(self.data)
            random.shuffle(data)
        else:
            data = self.data
        batch_data = []
        for ori_triplet in data:
            triplet_with_corrput = self.combine_with_corrupt(ori_triplet)
            for triplet in triplet_with_corrput:
                batch_data.append(triplet)
                if len(batch_data) == self.config.batch_size:
                    yield batch_data
                    batch_data.clear()
        if batch_data:
            yield batch_data

    def __len__(self):
        return len(self.data)




class Config(object):
    def __init__(self):
        self.corrupt_sample_size = 10  # Num of corrupted triplet generated.
        self.corrupt_sample_num = 1  # Num of corrupted used. Default 1 in paper.
        self.dataset = None  # DataSet
        self.batch_size = 10
        self.shuffle = False
        self.ent_total_num = 14951
        self.rel_total_num = 1345


# Something test

# file_path = '/Users/chenxiang/PycharmProjects/knowledge_graph_embedding/knowledge_graph_embedding/data/FB15k_ETL/train2id.txt'
# c = Config()
# datamanager_e = DataManagerE(file=file_path, config=c)

#
# dataset_e = TextLineDataset(file_path)
# iterator = dataset_e.make_one_shot_iterator()
# next_value = iterator.get_next()
#
# sess = tf.Session()
# print(sess.run(next_value))

