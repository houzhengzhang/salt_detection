#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : tf_utils.py
# @Time     : 2018/9/7 22:49 
# @Software : PyCharm
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
