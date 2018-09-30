#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : fb15k.py
__create_time__ : 2018/09/30
"""

import os
from collections import defaultdict

from project_basic_config import DATA_DIR

FB15K_DATA_DIR = os.path.join(DATA_DIR, 'FB15k')
FB15K_ETL_DATA_DIR = os.path.join(DATA_DIR, 'FB15k_ETL')


def write_object_into_file(obj, file_path):
    """
    Write object into file(allow dict and list only).
    :param obj:
    :param file_path:
    :return:
    """
    if isinstance(obj, dict):
        obj = ['%s\t%s' % (k, v) for k, v in obj.items()]
    assert isinstance(obj, list)

    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(file_path, 'w+') as w_file:
        w_file.writelines('\n'.join(obj))


def build_ent_rel_to_id(file_list):
    """
    Read each file in file_list and split line into <h,r,t> to get all entities and relations collection.
    Assign proper id to each entity and relation separately.
    :param file_list :
    :return : ent2id, rel2id
    """
    ent2id = defaultdict(int)
    rel2id = defaultdict(int)
    for file_name in file_list:
        file_path = os.path.join(FB15K_DATA_DIR, file_name)
        print(file_path)
        with open(file_path, 'r') as r_file:
            for line in r_file:
                h, r, t = line.strip().split('\t')
                if h not in ent2id:
                    ent2id[h] = len(ent2id)
                if r not in rel2id:
                    rel2id[r] = len(rel2id)
                if t not in ent2id:
                    ent2id[t] = len(ent2id)

    write_object_into_file(ent2id, os.path.join(FB15K_ETL_DATA_DIR, 'ent2id.txt'))
    write_object_into_file(rel2id, os.path.join(FB15K_ETL_DATA_DIR, 'rel2id.txt'))
    return ent2id, rel2id


def build_file_to_id(file_ori, ent2id, rel2id, file_dest):
    """
    Transfer file_ori from string triplet into id triplet.
    :param file_ori:
    :param ent2id:
    :param rel2id:
    :param file_dest:
    :return:
    """

    id_line_list = []
    with open(file_ori, 'r') as r_file:
        for line in r_file:
            h, r, t = line.strip().split('\t')
            h_id = ent2id[h]
            r_id = rel2id[r]
            t_id = ent2id[t]
            id_line_list.append('\t'.join(map(str, [h_id, r_id, t_id])))
    write_object_into_file(id_line_list, file_dest)


def main():
    fb15k_file_list = ['freebase_mtr100_mte100-test.txt', 'freebase_mtr100_mte100-train.txt',
                       'freebase_mtr100_mte100-valid.txt']
    fb15k_file_id_list = [file.split('.')[0].split('-')[-1] + '2id.txt' for file in fb15k_file_list]
    fb15k_file_list = [os.path.join(FB15K_DATA_DIR, file) for file in fb15k_file_list]
    fb15k_file_id_list = [os.path.join(FB15K_ETL_DATA_DIR, file) for file in fb15k_file_id_list]

    # Assign proper id for each entity and relation collected in files
    ent2id, rel2id = build_ent_rel_to_id(file_list=fb15k_file_list)
    # Transfer string triplet into id(int) triplet
    for file_ori, file_dest in zip(fb15k_file_list, fb15k_file_id_list):
        build_file_to_id(file_ori=file_ori, ent2id=ent2id, rel2id=rel2id, file_dest=file_dest)

if __name__ == '__main__':
    main()
