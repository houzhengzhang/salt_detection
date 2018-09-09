#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : data_generator.py
# @Time     : 2018/9/3 20:02 
# @Software : PyCharm
import tensorflow as tf


class DataGenerator:
    def __init__(self, config):
        self.__config = config
        self.__input_fn()

    # 解析 tfrecord
    def __parse_fn(self, example_proto):
        example_fmt = {"images": tf.FixedLenFeature([], tf.string),
                       "masks": tf.FixedLenFeature([], tf.string)}
        # {'coverage_class': <tf.Tensor 'ParseSingleExample/ParseSingleExample:0' shape=() dtype=int64>,
        # 'images': <tf.Tensor 'ParseSingleExample/ParseSingleExample:1' shape=() dtype=string>,
        # 'masks': <tf.Tensor 'ParseSingleExample/ParseSingleExample:2' shape=() dtype=string>}
        parsed_example = tf.parse_single_example(example_proto, example_fmt)

        image = tf.decode_raw(parsed_example["images"], tf.float64)
        image = tf.reshape(image, [101, 101])
        image = tf.expand_dims(image, -1)
        parsed_example["images"] = image

        masks = tf.decode_raw(parsed_example["masks"], tf.float64)
        masks = tf.reshape(masks, [101, 101])
        masks = tf.expand_dims(masks, -1)
        parsed_example["masks"] = masks

        # parsed_example["coverage_class"] = parsed_example["coverage_class"]
        return parsed_example

    def __input_fn(self):
        dataset = tf.data.TFRecordDataset(self.__config.train_tfrecord)
        # When possible, we recommend using the fused tf.contrib.data.shuffle_and_repeat transformation,
        # which combines the best of both worlds (good performance and strong ordering guarantees).
        # Otherwise, we recommend shuffling before repeating.
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=self.__config.buffer_size,
                                                                   count=self.__config.num_epochs))
        dataset = dataset.map(self.__parse_fn)
        dataset = dataset.batch(self.__config.batch_size)

        # 定义迭代器
        self.__next_element = dataset.make_one_shot_iterator().get_next()

    def next_batch(self):
        return self.__next_element


if __name__ == '__main__':
    data = DataGenerator(None)
