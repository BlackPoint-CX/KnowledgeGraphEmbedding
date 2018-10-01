#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : TransE.py
__create_time__ : 2018/09/27
"""
from progressbar import ProgressBar
from tensorflow.contrib.layers import xavier_initializer

from trans_base import TransBase
import tensorflow as tf


class TransE(TransBase):
    def __init__(self, config):
        super(TransE, self).__init__(config)

    def _cal_distance(self, h, r, t, norm_ord=1):
        """
        Calculate distance between h+r and t.
        :param h: Head Entity
        :param t: Tail Entity
        :param r: Relation
        :param norm_ord: Norm Order
        :return: Float
        """
        return tf.norm(h + r - t, ord=norm_ord)

    def build(self):
        self.add_placeholder_op()
        self.add_embedding_op()

    def add_placeholder_op(self):
        """ Add placeholders for original triplet (h,r,t) and corrupted triplet( h',r',t') respectively. """
        self.batch_ori_h = tf.placeholder(dtype=tf.int64, shape=[None],
                                          name='batch_h')  # batch input of ori head entity
        self.batch_ori_r = tf.placeholder(dtype=tf.int64, shape=[None],
                                          name='batch_r')  # batch input of ori relation
        self.batch_ori_t = tf.placeholder(dtype=tf.int64, shape=[None],
                                          name='batch_t')  # batch input of ori tail entity

        self.batch_cor_h = tf.placeholder(dtype=tf.int64, shape=[None],
                                          name='batch_h')  # batch input of corrupt head entity
        self.batch_cor_r = tf.placeholder(dtype=tf.int64, shape=[None],
                                          name='batch_r')  # batch input of corrupt relation
        self.batch_cor_t = tf.placeholder(dtype=tf.int64, shape=[None],
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

        self.batch_corrput_h_embedding = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.batch_cor_h,
                                                                name='batch_corrput_h_embedding')
        self.batch_corrupt_r_embedding = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.batch_cor_r,
                                                                name='batch_corrupt_r_embedding')
        self.batch_corrupt_t_embedding = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.batch_cor_t,
                                                                name='batch_corrupt_t_embedding')

    def add_loss_op(self):
        _p_sore = self._cal_distance(self.batch_ori_h_embedding, self.batch_ori_r_embedding, self.batch_ori_t_embedding)
        _n_score = self._cal_distance(self.batch_corrput_h_embedding, self.batch_corrupt_r_embedding,
                                      self.batch_corrupt_t_embedding)
        self.loss = tf.reduce_sum(tf.maximum(_p_sore + self.config.margin - _n_score, 0))

    def add_score_op(self, h, r, t):
        """ Calculate dissimilarity between h,r and t """
        h_embedding = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=h)
        r_embedding = tf.nn.embedding_lookup(params=self.rel_embeddings, ids=r)
        t_embedding = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=t)
        self.score = tf.matmul(tf.transpose(h_embedding), t_embedding) + tf.matmul(tf.transpose(r_embedding),
                                                                                   (t_embedding - h_embedding))

    def run_single_epoch(self, train, val, epoch_idx):
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        pgb = ProgressBar(maxval=nbatches).start()
        for batch_index, records in enumerate(train):
            feed_dict = self.get_feed_dict(records=records)
            _, loss, summary = self.sess.run(fetches=[self.train_op, self.loss, self.merged], feed_dict=feed_dict)
            pgb.update(batch_index)

            if batch_index % 10 == 0:
                self.file_writer.add_summary(summary=summary, global_step=epoch_idx * batch_size + batch_index)

        metrics = self.evaluate(val)
        msg = '-'.join(['{}:{:04.2f}'.format(k, v) for k, v in metrics.items()])
        self.logger.info(msg)
        return metrics

    def get_feed_dict(self, records):
        ori_h_list, ori_r_list, ori_t_list, cor_h_list, cor_r_list, cor_t_list = [], [], [], [], [], []
        for record in records:
            ((ori_h, ori_r, ori_t), (cor_h, cor_r, cor_t)) = record
            ori_h_list.append(ori_h)
            ori_r_list.append(ori_r)
            ori_t_list.append(ori_t)
            cor_h_list.append(cor_h)
            cor_r_list.append(cor_r)
            cor_t_list.append(cor_t)
        feed_dict = {
            self.batch_ori_h: ori_h_list,
            self.batch_ori_r: ori_r_list,
            self.batch_ori_t: ori_t_list,
            self.batch_cor_h: cor_h_list,
            self.batch_cor_r: cor_h_list,
            self.batch_cor_t: cor_h_list

        }
        return feed_dict

    def add_predict_op(self):
        pass
