#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : read_tfrecord.py
# @Time     : 2018/9/3 16:18 
# @Software : PyCharm
import tensorflow as tf


def read_records(tfrecord_file, batch_size, one_hot, n_classes):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([tfrecord_file], shuffle=False)
        # init TFRecordReader class
        reader = tf.TFRecordReader()
        key, values = reader.read(filename_queue)  # filename_queue
        # parse_single_example将Example协议内存块(protocol buffer)解析为张量
        # feature = {'images': _bytes_feature(img_data),
        #            'masks': _bytes_feature(mask_data),
        #            'coverage_class': _int64_feature(int(getattr(row, "coverage_class")))}
        features = tf.parse_single_example(values,
                                           features={
                                               'images': tf.FixedLenFeature([], tf.string),
                                               'masks': tf.FixedLenFeature([], tf.string),
                                               'coverage_class': tf.FixedLenFeature([], tf.int64)
                                           })
        # decode
        image = tf.decode_raw(features['images'], tf.float64)
        image = tf.reshape(image, [128, 128])
        image = tf.expand_dims(image, -1)
        # decode
        masks = tf.decode_raw(features['masks'], tf.float64)
        masks = tf.reshape(masks, [128, 128])
        masks = tf.expand_dims(masks, -1)
        # label
        coverage_class = tf.cast(features['coverage_class'], tf.int32)

        image_batch, mask_batch, label_batch = tf.train.shuffle_batch([image, masks, coverage_class],
                                                                      batch_size=batch_size,
                                                                      num_threads=1,
                                                                      capacity=1500000,
                                                                      min_after_dequeue=512)
        # one hot
        if one_hot:
            label_batch = tf.one_hot(label_batch, depth=n_classes)
            # tf.one_hot之后label的类型变为tf.float32，后面运行会出bug
            # 所以在这里再次调用tf.cast
            label_batch = tf.cast(label_batch, tf.int32)
    return image_batch, mask_batch, label_batch


if __name__ == '__main__':
    image_batch, mask_batch, label_batch = read_records('train.tfrecords', batch_size=1, one_hot=True, n_classes=11)
    coord = tf.train.Coordinator()
    sess = tf.Session()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while not coord.should_stop():
            image, mask, label = sess.run([image_batch, mask_batch, label_batch])
            print(image, image.shape)
            exit()

    except tf.errors.OutOfRangeError:
        print("Done!")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
