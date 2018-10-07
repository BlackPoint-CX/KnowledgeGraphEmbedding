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
        self.logger = config.logger
        self.sess = None
        self.saver = None

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
        self.saver.save(self.sess, self.config.model_dir)

    def close_session_op(self):
        """ Close session. """
        self.logger.info('Close Session.')
        self.sess.close()

    def restore_session_op(self, model_path):
        """ Restore session from input model path. """
        self.logger.info('Restore session from input <model path>.')
        self.saver.restore(self.sess, model_path)

    def train_placeholder_op(self):
        pass

    def train_loss_op(self):
        pass

    def train_embedding_op(self):
        pass

    def add_train_op(self, loss):
        with tf.variable_scope('train_step'):
            _optimizer = self.config.optimizer.lower()

            if _optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)
            elif _optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer()
            elif _optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=self.config.lr)
            elif _optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.lr)
            else:
                raise NotImplementedError('Optimizer should be choosen in [sgd, adam, adagrad , rmsprop]')

            if self.config.clip > 0:
                gradients, variables = zip(*optimizer.compute_gradients(loss=loss))
                gradients, gradients_norm = tf.clip_by_global_norm(gradients, self.config.clip)
                self.train_op = optimizer.apply_gradients(grads_and_vars=zip(gradients, variables))
            else:
                self.train_op = optimizer.minimize(loss)

    def add_summary_op(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(logdir=self.config.summary_dir, graph=self.sess.graph)

    def train(self, train, val):
        best_score = self.config.best_score
        epochs_no_impv = self.config.epochs_no_impv
        self.add_summary_op()

        for epoch_idx in range(self.config.epochs):
            self.logger.info('Processing epoch %d' % epoch_idx)
            metrics = self.run_epoch(train, val, epoch_idx)
            self.config.lr *= self.config.lr_decay

            if isinstance(metrics, dict):
                pass
            else:
                if metrics > best_score:
                    epochs_no_impv = 0
                    self.save_session_op()
                    best_score = metrics
                    self.logger.info('Achieving New Best Score : %d' % best_score)
                else:
                    epochs_no_impv += 1
                    if epochs_no_impv > self.config.epochs_no_impv:
                        self.logger.info('Stop after %d epochs with out improvement.' % epochs_no_impv)
                        break

