#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : TransBase.py
__create_time__ : 2018/09/27
"""
import tensorflow as tf


class TransBase(object):
    def __init__(self, config):
        self.config = config
        self.logger = None
        self.sess = None
        self.saver = None

    @property
    def config(self):
        return self.config

    @config.setter
    def config(self, config):
        self.config = config

    def init_session_op(self):
        """ Initialize model's session and relate global variables. """
        self.logger.info('Initialize Session and Global Variables.')
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_session_op(self):
        """ Save session. """
        self.logger.info('Save Session.')
        if not tf.gfile.Exists(self.config.model_dir):
            tf.gfile.MakeDirs(self.config.model_dir)
        self.saver(self.sess, self.config.model_dir)

    def close_session_op(self):
        """ Close session. """
        self.logger.info('Close Session.')
        self.sess.close()

    def restore_session_op(self, model_path):
        """ Restore session from input model path. """
        self.logger.info('Restore session from input <model path>.')
        self.saver.restore(self.sess, model_path)

    def add_placeholder_op(self):
        pass

    def add_loss_op(self):
        pass

    def add_embedding_op(self):
        pass

    def add_train_op(self):
        with tf.variable_scope('train_step'):
            _optimizer = self.config.optimizer.lower()

            if _optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer()
            elif _optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer()
            elif _optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer()
            elif _optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer()
            else:
                raise NotImplementedError('Optimizer should be choosen in [sgd, adam, adagrad , rmsprop]')

            if self.config.clip > 0:
                gradients, variables = zip(*optimizer.compute_gradients(loss=self.loss))
                gradients, gradients_norm = tf.clip_by_global_norm(gradients, self.config.clip)
                self.train_op = optimizer.apply_gradients(grads_and_vars=zip(gradients, variables))
            else:
                self.train_op = optimizer.minimize(self.loss)

    def add_summary_op(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(logdir=self.config.summary_dir, graph=self.sess.graph)

    def train(self, train, dev):
        best_score = 0
        epochs_no_impv = self.config.epochs_no_impv
        self.add_summary_op()

        for epoch_idx in range(self.config.epochs):
            self.logger.info()
            score = self.run_single_epoch(train, dev, epoch_idx)
            self.config.lr *= self.config.lr_decay
            if score > best_score:
                epochs_no_impv = 0
                self.save_session_op()
                best_score = score
                self.logger.info()
            else:
                epochs_no_impv += 1
                if epochs_no_impv > self.config.epochs_no_impv:
                    self.logger.info()
                    break

    def evaluate(self, val):
        total_rank = 0
        total_hits_at_10 = 0
        val_len = len(val)
        for records in val:
            for record in records:
                score_list = []
                ((ori_h, ori_r, ori_t), (cor_h, cor_r, cor_t)) = record
                for i in range(self.config.ent_total_num):
                    if i != ori_h:
                        cor_score = self.score(i, ori_r, ori_t)
                        score_list.append(('cor', cor_score))
                ori_score = self.score(ori_h, ori_r, ori_t)
                score_list.append(('ori', ori_score))
                sorted_score_list = sorted(score_list, key=lambda r: r[1], reverse=True)
                index = sorted_score_list.index(('ori', ori_score)) + 1  # index start from i since hits@10 start with 1
                total_rank += index
                if index <= 10:
                    total_hits_at_10 += 1
        mean_rank = total_rank / val_len
        hits_at_10 = total_hits_at_10 / val_len
        return {'MeanRank': mean_rank, 'Hits@10': hits_at_10}
