#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : tfrecord_dataset.py
# @Time     : 2018/9/3 17:12 
# @Software : PyCharm
import tensorflow as tf


# 解析 tfrecord
def parse_fn(example_proto):
    example_fmt = {"images": tf.FixedLenFeature([], tf.string),
                   "masks": tf.FixedLenFeature([], tf.string),
                   "coverage_class": tf.FixedLenFeature([], tf.int64)}
    # {'coverage_class': <tf.Tensor 'ParseSingleExample/ParseSingleExample:0' shape=() dtype=int64>,
    # 'images': <tf.Tensor 'ParseSingleExample/ParseSingleExample:1' shape=() dtype=string>,
    # 'masks': <tf.Tensor 'ParseSingleExample/ParseSingleExample:2' shape=() dtype=string>}
    parsed_example = tf.parse_single_example(example_proto, example_fmt)

    image = tf.decode_raw(parsed_example["images"], tf.float64)
    image = tf.reshape(image, [128, 128])
    image = tf.expand_dims(image, -1)
    parsed_example["images"] = image

    masks = tf.decode_raw(parsed_example["masks"], tf.float64)
    masks = tf.reshape(masks, [128, 128])
    masks = tf.expand_dims(masks, -1)
    parsed_example["masks"] = masks

    parsed_example["coverage_class"] = parsed_example["coverage_class"]
    return parsed_example


def input_fn(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    # When possible, we recommend using the fused tf.contrib.data.shuffle_and_repeat transformation,
    # which combines the best of both worlds (good performance and strong ordering guarantees).
    # Otherwise, we recommend shuffling before repeating.
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000, count=3))
    dataset = dataset.map(parse_fn)
    dataset = dataset.batch(8)
    return dataset


dataset = input_fn("train.tfrecords")
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.InteractiveSession()

i = 0
while True:
    try:
        example = sess.run(next_element)
        print(example)
        i += 1
    except tf.errors.OutOfRangeError:
        print("Done")
        print(i)
        break
