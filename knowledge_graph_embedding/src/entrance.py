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

FLAGS = None


def arg_parser_build(arg_parser):
    arg_parser.add_argument(name_or_flags='--mode', type=str, default='trans_e', choices=['trans_e'],
                            help='Choose proper embedding mode.')
    arg_parser.add_argument(name_or_flats='', type=str, default='', choices=[], help='')
    arg_parser.add_argument(name_or_flats='', type=str, default='', choices=[], help='')
    arg_parser.add_argument(name_or_flats='', type=str, default='', choices=[], help='')
    arg_parser.add_argument(name_or_flats='', type=str, default='', choices=[], help='')



def hparams_builder(flags):
    return HParams(
        mode=flags.mode
    )


def main(unparsed):
    hparams = hparams_builder(FLAGS)
    if hparams.mode == 'trans_e':
        pass


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser_build(arg_parser)
    FLAGS, unparsed = arg_parser.parse_known_args()
    tf.app.run(main=main, argv=unparsed)
