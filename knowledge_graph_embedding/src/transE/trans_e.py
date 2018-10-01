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

    def _cal_score(self, h, r, t):
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
        self.add_embedding_op()

    def add_placeholder_op(self):
        """ Add placoholders """
        self.batch_ori_h = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_seq_size],
                                          name='batch_h')  # batch input of ori head entity
        self.batch_ori_r = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_seq_size],
                                          name='batch_r')  # batch input of ori relation
        self.batch_ori_t = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_seq_size],
                                          name='batch_t')  # batch input of ori tail entity

        self.batch_corrupt_h = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_seq_size],
                                              name='batch_h')  # batch input of corrupt head entity
        self.batch_corrput_r = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_seq_size],
                                              name='batch_r')  # batch input of corrupt relation
        self.batch_corrput_t = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_seq_size],
                                              name='batch_t')  # batch input of corrput entity

    def add_embedding_op(self):
        """
        Initializer embedding table for entity and relation
        """
        self.ent_embeddings = tf.get_variable(name='ent_embedding', dtype=tf.float32,
                                              shape=[self.config.ent_total_num, self.config.embedding_dim],
                                              initializer=xavier_initializer(uniform=True)
                                              )
        self.rel_embeddings = tf.get_variable(name='rel_embedding', dtype=tf.float32,
                                              shape=[self.config.rel_total_num, self.config.embedding_dim],
                                              initializer=xavier_initializer(uniform=True))

    def add_trans_embedding_op(self):
        self.batch_ori_h_embedding = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.batch_ori_h,
                                                            name='batch_h_embedding')
        self.batch_ori_r_embedding = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.batch_ori_r,
                                                            name='batch_r_embedding')
        self.batch_ori_t_embedding = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.batch_ori_t,
                                                            name='batch_t_embedding')

        self.batch_corrput_h_embedding = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.batch_corrupt_h,
                                                                name='batch_corrput_h_embedding')
        self.batch_corrupt_r_embedding = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.batch_corrput_r,
                                                                name='batch_corrupt_r_embedding')
        self.batch_corrupt_t_embedding = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.batch_corrput_t,
                                                                name='batch_corrupt_t_embedding')

    def add_loss_op(self):
        _p_sore = self._cal_score(self.batch_ori_h_embedding,self.batch_ori_r_embedding,self.batch_ori_t_embedding)
        _n_score = self._cal_score(self.batch_corrput_h_embedding,self.batch_corrupt_r_embedding,self.batch_corrupt_t_embedding)
        self.loss = tf.reduce_sum(tf.maximum(_p_sore + self.config.margin - _n_score,0))


    def add_predict_op(self):
        pass




