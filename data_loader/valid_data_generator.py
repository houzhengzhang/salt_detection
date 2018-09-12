#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : valid_data_generator.py
# @Time     : 2018/9/8 20:03 
# @Software : PyCharm
import tensorflow as tf


class ValidDataGenerator:
    def __init__(self, config):
        self.__config = config
        self.__input_fn()

    def __parse_fn(self, example_proto):
        example_fmt = {"images": tf.FixedLenFeature([], tf.string),
                       "masks": tf.FixedLenFeature([], tf.string)}
        parsed_example = tf.parse_single_example(example_proto, example_fmt)

        image = tf.decode_raw(parsed_example["images"], tf.float64)
        image = tf.reshape(image, [128, 128])
        image = tf.expand_dims(image, -1)
        parsed_example["images"] = image

        masks = tf.decode_raw(parsed_example["masks"], tf.float64)
        masks = tf.reshape(masks, [128, 128])
        masks = tf.expand_dims(masks, -1)
        parsed_example["masks"] = masks

        return parsed_example

    def __input_fn(self):
        dataset = tf.data.TFRecordDataset(self.__config.valid_tfrecord)
        dataset = dataset.map(self.__parse_fn)
        dataset = dataset.batch(self.__config.batch_size)

        # 定义迭代器
        self.__iterator = dataset.make_initializable_iterator()

    def get_iterator(self):
        return self.__iterator
