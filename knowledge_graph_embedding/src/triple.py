#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : triple.py
__create_time__ : 2018/09/29
"""


class Triple(object):
    def __init__(self, h, t, r):
        """
        :param h: Head Entity
        :param t: Tail Entity
        :param r: Relation
        """
        self.h = h
        self.t = t
        self.r = r

    @property
    def h(self):
        return self.h

    @h.setter
    def h(self, h):
        self.h = h

    @property
    def r(self):
        return self.r

    @r.setter
    def r(self, r):
        self.r = r

    @property
    def t(self):
        return self.t

    @t.setter
    def t(self, t):
        self.t = t
