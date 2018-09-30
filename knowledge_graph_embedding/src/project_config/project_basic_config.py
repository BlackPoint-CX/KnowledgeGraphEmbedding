#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : project_config.py
__create_time__ : 2018/09/30
"""
from configparser import ConfigParser
import os
from pprint import pprint

config_parser = ConfigParser()
current_dir = os.path.dirname(os.path.abspath(__file__))
config_parser.read(os.path.join(current_dir, 'basic.config'))
PROJECT_DIR = config_parser.get('PROJECT_CONFIG', 'PROJECT_dir')
DATA_DIR = os.path.join(PROJECT_DIR, 'knowledge_graph_embedding/data')
SRC_DIR = os.path.join(PROJECT_DIR, 'knowledge_graph_embedding/src')
SCRIPT_DIR = os.path.join(PROJECT_DIR, 'knowledge_graph_embedding/script')
LOG_DIR = os.path.join(PROJECT_DIR, 'knowledge_graph_embedding/log')
TEST_DIR = os.path.join(PROJECT_DIR, 'tests')

config_dict = {'current_dir': current_dir, 'PROJECT_DIR': PROJECT_DIR, 'DATA_DIR': DATA_DIR,
               'SRC_DIR': SRC_DIR, 'SCRIPT_DIR': SCRIPT_DIR, 'LOG_DIR': LOG_DIR, 'TEST_DIR': TEST_DIR}

for k, v in config_dict.items():
    print('%s -> %s' % (k, v))
