#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : TransE.py
__create_time__ : 2018/09/27
"""
from tensorflow.contrib.layers import xavier_initializer

from trans_base import TransBase
import tensorflow as tf


class TransE(TransBase):
    def __init__(self, config):
        super(TransE, self).__init__(config)

    def _cal_score(self, h, t, r):
        """
        Calculate relate score based on given h, t and r.
        :param h: Head Entity
        :param t: Tail Entity
        :param r: Relation
        :return: Float
        """
        return abs(h + r - t)

    def build(self):
        self.add_placeholder_op()

    def add_embedding_op(self):
        """
        Initializer
        :return:
        """
        self.ent_embedding = tf.get_variable(name='ent_embedding', dtype=tf.float32,
                                             shape=[self.config.ent_total_num, self.config.hidden_size],
                                             initializer=xavier_initializer(uniform=True)
                                             )
        self.rel_embedding = tf.get_variable(name='rel_embedding', dtype=tf.float32,
                                             shape=[self.config.rel_total_num, self.config.hidden_size],
                                             initializer=xavier_initializer(uniform=True))

    def add_loss_op(self):

        pass
