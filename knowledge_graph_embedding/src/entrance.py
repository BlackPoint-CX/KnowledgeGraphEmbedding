#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : entrance.py
__create_time__ : 2018/09/27
"""
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.contrib.training import HParams

from trans_e import TransE
from trans_e_dataset import DataManagerE

FLAGS = None


def arg_parser_build(arg_parser):
    arg_parser.add_argument(name_or_flags='--mode', type=str, default='trans_e', choices=['trans_e'],
                            help='Choose proper embedding mode.')
    arg_parser.add_argument(name_or_flats='--train_file_path', type=str, default='', choices=[], help='')
    arg_parser.add_argument(name_or_flats='--test_file_path', type=str, default='', choices=[], help='')
    arg_parser.add_argument(name_or_flats='--dev_file_path', type=str, default='', choices=[], help='')
    arg_parser.add_argument(name_or_flats='--load_hparams_json', type=str, default='', choices=[], help='')


def hparams_builder(flags):
    return HParams(
        mode=flags.mode
    )


def main(unparsed):
    hparams = hparams_builder(FLAGS)
    if hparams.mode == 'trans_e':
        train_data_manager_e = DataManagerE(file=hparams.train_file_path, config=hparams)
        # test_data_manager_e = DataManagerE(file=hparams.test_file_path, config=hparams)
        dev_data_manager_e = DataManagerE(file=hparams.dev_file_path, config=hparams)

        model = TransE(config=hparams)
        model.build()
        model.train(train=train_data_manager_e, dev=dev_data_manager_e)
        model.save_session_op()

    else:
        raise NotImplementedError


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser_build(arg_parser)
    FLAGS, unparsed = arg_parser.parse_known_args()
    tf.app.run(main=main, argv=unparsed)
