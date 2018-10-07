#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : TransE.py
__create_time__ : 2018/09/27
"""
import timeit

from progressbar import ProgressBar
import numpy as np
from trans_base import TransBase
import tensorflow as tf
import math
import multiprocessing as mp


class TransE(TransBase):
    def __init__(self, config):
        super(TransE, self).__init__(config)
        self.n_rank_calculator = 8

    def build_train_graph(self):
        self.train_placeholder_op()
        self.train_embedding_op()
        self.train_trans_embedding_op()
        self.train_loss_op()
        self.add_train_op(loss=self.train_loss)

        self.eval_placeholder()
        self.eval_single_triplet()
        self.init_session_op()

    def train_placeholder_op(self):
        """ Add placeholders for original triplet (h,r,t) and corrupted triplet( h',r',t') respectively. """
        with tf.variable_scope('train_placeholder_op'):
            self.pos_triplet = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='pos_triplet')
            self.neg_triplet = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='neg_triplet')

    def train_embedding_op(self):
        """ Initialize and normalize embeddings for entities and relations. """
        with tf.variable_scope('train_embedding_init'):
            bound = 6 / math.sqrt(self.config.embedding_dim)
            self.ent_embeddings = tf.get_variable(name='ent_embedding', dtype=tf.float32,
                                                  shape=[self.config.ent_total_num, self.config.embedding_dim],
                                                  initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound)
                                                  )
            self.rel_embeddings = tf.get_variable(name='rel_embedding', dtype=tf.float32,
                                                  shape=[self.config.rel_total_num, self.config.embedding_dim],
                                                  initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound)
                                                  )

            self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
            self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

    def train_trans_embedding_op(self):
        with tf.variable_scope('train_embedding_lookup'):
            self.pos_head = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.pos_triplet[:, 0])
            self.pos_rel = tf.nn.embedding_lookup(params=self.rel_embeddings, ids=self.pos_triplet[:, 1])
            self.pos_tail = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.pos_triplet[:, 2])
            self.neg_head = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.neg_triplet[:, 0])
            self.neg_rel = tf.nn.embedding_lookup(params=self.rel_embeddings, ids=self.neg_triplet[:, 1])
            self.neg_tail = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.neg_triplet[:, 2])

    def train_loss_op(self):
        with tf.variable_scope('train_loss'):
            pos_dist = self.pos_head + self.pos_rel - self.pos_tail
            neg_dist = self.neg_head + self.neg_rel - self.neg_tail

            _pos_score = tf.reduce_sum(tf.abs(pos_dist), axis=1)
            _neg_score = tf.reduce_sum(tf.abs(neg_dist), axis=1)

            self.train_loss = tf.reduce_sum(tf.maximum(_pos_score + self.config.margin - _neg_score, 0),
                                            name='train_loss')
            tf.summary.scalar(name='train_loss', tensor=self.train_loss)

    def run_epoch(self, train, val, epoch_idx):
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size  # Total number of batches

        input_queue = mp.Queue()
        for batch_index, records in enumerate(train):

            if batch_index % 10000 == 0:
                self.logger.info('Processing %d batch' % batch_index)

            train_feed_dict = self.train_feed_dict(records=records)
            _, loss, summary = self.sess.run(fetches=[self.train_op, self.train_loss, self.merged],
                                             feed_dict=train_feed_dict)

            if batch_index % 10 == 0:
                self.file_writer.add_summary(summary=summary, global_step=epoch_idx * batch_size + batch_index)

        self.evaluate(val)

        return {}

    def train_feed_dict(self, records):
        """ Initialize feed dict for training """
        batch_pos_triplet, batch_neg_triplet = [], []
        for record in records:
            pos_triplet, neg_triplet = record
            batch_pos_triplet.append(pos_triplet)
            batch_neg_triplet.append(neg_triplet)
        feed_dict = {
            self.pos_triplet: batch_pos_triplet,
            self.neg_triplet: batch_neg_triplet

        }
        return feed_dict

    def eval_single_triplet(self):
        with tf.name_scope('embedding_lookup'):
            head = tf.nn.embedding_lookup(self.ent_embeddings, self.eval_triplet[0])
            relation = tf.nn.embedding_lookup(self.rel_embeddings, self.eval_triplet[1])
            tail = tf.nn.embedding_lookup(self.ent_embeddings, self.eval_triplet[2])

        with tf.name_scope('link'):
            distance_head_prediction = self.ent_embeddings + relation - tail
            distance_tail_prediction = head + relation - self.ent_embeddings

        with tf.name_scope('rank'):
            _, self.idx_head_prediction = tf.nn.top_k(input=tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                      k=self.config.ent_total_num, sorted=True)
            _, self.idx_tail_prediction = tf.nn.top_k(input=tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                      k=self.config.ent_total_num, sorted=True)

    def eval_calculate_rank(self, in_queue, out_queue):
        while True:
            idx_prediction = in_queue.get()
            if idx_prediction is None:
                in_queue.task_done()
            else:
                eval_triplet, idx_head_prediction, idx_tail_prediction = idx_prediction
                head, relation, tail = eval_triplet
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0

                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, relation, tail) in []:
                            continue
                        else:
                            head_rank_filter += 1

                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, relation, candidate) in []:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def eval_placeholder(self):
        self.eval_triplet = tf.placeholder(dtype=tf.int32, shape=[3], name='eval_triplet')

    def evaluate(self, val):
        msg = '-' * 30 + 'start evaluation on validation dataset' + '-' * 30
        self.logger.info(msg)
        start = timeit.default_timer()
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()

        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.eval_calculate_rank,
                       kwargs={'in_queue': eval_result_queue, 'out_queue': rank_result_queue}).start()

        val_len = len(val)
        n_used_eval_triple = 0
        for batch in val:
            for record in batch:
                eval_triplet, _ = record
                idx_head_prediction, idx_tail_prediction = self.sess.run(
                    fetches=[self.idx_head_prediction, self.idx_tail_prediction],
                    feed_dict={self.eval_triplet: eval_triplet})
                eval_result_queue.put((eval_triplet, idx_head_prediction, idx_tail_prediction))
                n_used_eval_triple += 1
                print(
                    '[{:.3f}s] # evaluation triple : {}/{}'.format(timeit.default_timer() - start,
                                                                   n_used_eval_triple,
                                                                   val_len), end='\r')
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)

        print('-' * 30, 'Join all rank calculator', '-' * 30)
        eval_result_queue.join()
        print('-' * 30, 'All rank calculation completed', '-' * 30)
        print('-' * 30, 'Obtaining evaluation results', '-' * 30)

        ''' Raw '''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0

        ''' Filter '''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0

        for _ in range(val_len):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10: head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10: tail_hits10_raw += 1

            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10: head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10: tail_hits10_filter += 1

        head_meanrank_raw /= val_len
        head_hits10_raw /= val_len
        tail_meanrank_raw /= val_len
        tail_hits10_raw /= val_len

        head_meanrank_filter /= val_len
        head_hits10_filter /= val_len
        tail_meanrank_filter /= val_len
        tail_hits10_filter /= val_len

        print('-' * 30, 'Raw', '-' * 30)
        print('Head Prediction : MeanRank : {:.2f}, Hits@10 : {:.2f}'.format(head_meanrank_raw, head_hits10_raw))
        print('Tail Prediction : MeanRank : {:.2f}, Hits@10 : {:.2f}'.format(tail_meanrank_raw, tail_hits10_raw))

        print('-' * 30, 'Filter', '-' * 30)
        print('Head Prediction : MeanRank : {:.2f}, Hits@10 : {:.2f}'.format(head_meanrank_filter, head_hits10_filter))
        print('Tail Prediction : MeanRank : {:.2f}, Hits@10 : {:.2f}'.format(tail_meanrank_filter, tail_hits10_filter))
        return {}
