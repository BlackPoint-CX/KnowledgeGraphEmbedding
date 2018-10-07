#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : entrance.py
__create_time__ : 2018/09/27
"""
import os
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.contrib.training import HParams

from common_funcs import build_logger
from project_basic_config import DATA_DIR, SUMMARY_DIR, LOG_DIR, MODEL_DIR
from trans_e import TransE
from trans_e_dataset import DataManagerE

FLAGS = None
TRAIN_FILE_PATH = os.path.join(DATA_DIR, 'FB15k_ETL/train2id.txt')
TEST_FILE_PATH = os.path.join(DATA_DIR, 'FB15k_ETL/test2id.txt')
VALID_FILE_PATH = os.path.join(DATA_DIR, 'FB15k_ETL/valid2id.txt')
LOG_FILE_PATH = os.path.join(LOG_DIR, 'entrance.log')


def arg_parser_build(arg_parser):
    arg_parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    arg_parser.add_argument('--clip', type=int, default=-1, help='Clip')
    arg_parser.add_argument('--corrupt_sample_num', type=int, default=1,
                            help='#Num of chosen corrupted triplets')
    arg_parser.add_argument('--corrupt_sample_size', type=int, default=2,
                            help='#Num of generated corrupted triplets')
    arg_parser.add_argument('--embedding_dim', type=int, default=50, help='Dimension of embedding')
    arg_parser.add_argument('--epochs', type=int, default=20, help='Running epochs')
    arg_parser.add_argument('--epochs_no_impv', type=int, default=3, help='Epochs without improvements')
    arg_parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    arg_parser.add_argument('--lr_decay', type=float, default=1, help='Decay rate of learning rate ')
    arg_parser.add_argument('--margin', type=float, default=1,
                            help='Margin between original triplet and corrupted triplet')
    arg_parser.add_argument('--mode', type=str, default='trans_e', choices=['trans_e'],
                            help='Choose proper embedding mode.')
    arg_parser.add_argument('--model_dir', type=str, default=MODEL_DIR, help='Directory for saving model.')
    arg_parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adagrad', 'rmsprop'],
                            help='Optimizer name')
    arg_parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle data')
    arg_parser.add_argument('--summary_dir', type=str, default=SUMMARY_DIR, help='Directory for store summary')
    arg_parser.add_argument('--test_file_path', type=str, default=TEST_FILE_PATH, choices=[], help='')
    arg_parser.add_argument('--train_file_path', type=str, default=TRAIN_FILE_PATH, choices=[], help='')
    arg_parser.add_argument('--val_file_path', type=str, default=VALID_FILE_PATH, choices=[], help='')
    arg_parser.add_argument('--load_hparams_json', type=str, default=None, help='')


def hparams_builder(flags):
    return HParams(
        batch_size=flags.batch_size,
        best_score={'MeanRank': 0, 'Hits@10': 0},
        clip=flags.clip,
        corrupt_sample_num=1,
        corrupt_sample_size=2,
        embedding_dim=flags.embedding_dim,
        ent_total_num=14951,
        epochs=flags.epochs,
        epochs_no_impv=flags.epochs_no_impv,
        lr=flags.lr,
        lr_decay=flags.lr_decay,
        logger=build_logger(LOG_FILE_PATH),
        margin=flags.margin,
        mode=flags.mode,
        model_dir=flags.model_dir,
        optimizer=flags.optimizer,
        rel_total_num=1345,
        summary_dir=flags.summary_dir,
        shuffle=flags.shuffle,
        test_file_path=flags.test_file_path,
        train_file_path=flags.train_file_path,
        val_file_path=flags.val_file_path,

    )


def main(FLAGS):
    hparams = hparams_builder(FLAGS)
    if hparams.mode == 'trans_e':
        train_data_manager_e = DataManagerE(file=hparams.train_file_path, config=hparams)
        val_data_manager_e = DataManagerE(file=hparams.val_file_path, config=hparams)
        for file in os.listdir(hparams.summary_dir):
            os.remove(os.path.join(hparams.summary_dir, file))

        model = TransE(config=hparams)
        model.build_train_graph()
        model.train(train=train_data_manager_e, val=val_data_manager_e)
        model.save_session_op()
        metrics = model.evaluate(val=val_data_manager_e)
        msg = '-'.join(['{}:{:04.2f}'.format(k, v) for k, v in metrics.items()])
        model.logger.info(msg)
        print(msg)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser_build(arg_parser)
    FLAGS, unparsed = arg_parser.parse_known_args()
    # tf.app.run(main=main, argv=FLAGS)
    main(FLAGS)
