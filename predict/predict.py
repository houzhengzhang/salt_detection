#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : predict.py
# @Time     : 2018/9/7 14:16 
# @Software : PyCharm
import tensorflow as tf
from keras.preprocessing.image import load_img
import numpy as np
from tqdm import tqdm
import pandas as pd
from skimage.transform import resize
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from models.iou_metric import iou_metric_batch
from layers.layers_fcn_gcn import conv_module, global_conv_module, boundary_refine, deconv_module


class Model:
    def __init__(self, checkpoint_dir, graph_name):
        self.checkpoint_dir = checkpoint_dir
        self.graph_name = graph_name
        self.load_weight()
        print("weight load done!")

    def load_weight(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        # 获取默认图
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.80  # 占用GPU90%的显存
        self.graph = tf.get_default_graph()
        # self.is_training = tf.placeholder(tf.bool, name="is_training")
        # self.x = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])

        # self.fcn_model = self.build_fcn_net(self.x, self.is_training)
        self.sess = tf.Session(config=tf_config)
        # self.saver = tf.train.Saver()
        self.saver = tf.train.import_meta_graph(self.graph_name)
        # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # self.sess.run(init)
        self.saver.restore(self.sess, latest_checkpoint)

        # g_list = self.graph.get_operations()
        # for g in g_list:
        #     print(g.name)

        # ********************************* #
        # bug  Placeholder 1#
        # ********************************* #
        self.input_op = self.graph.get_operation_by_name("Placeholder").outputs[0]
        # self.logits = self.graph.get_operation_by_name("logits/Conv2D").outputs[0]
        self.logits = self.graph.get_operation_by_name("logits/Conv2D").outputs[0]
        self.is_training = self.graph.get_operation_by_name("is_training").outputs[0]

    def predict(self, input):
        # logits = self.sess.run(self.fcn_model, feed_dict={self.x: input, self.is_training: False})
        logits = self.sess.run(self.logits, feed_dict={self.input_op: input, self.is_training: False})
        return logits


def rlenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    img = np.squeeze(img)
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


img_size_ori = 101
img_size_target = 128


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def read_valid():
    def parse_fn(example_proto):
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

    def input_fn(filenames):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parse_fn)
        return dataset

    dataset = input_fn("../input/valid.tfrecords")
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    sess = tf.InteractiveSession()
    images = []
    masks = []
    while True:
        try:
            example = sess.run(next_element)
            image = example["images"]
            mask = example["masks"]
            images.append(image)
            masks.append(mask)
        except tf.errors.OutOfRangeError:
            return np.array(images), np.array(masks)


def compute_thresholds(model):
    valid_image, valid_mask = read_valid()
    preds_valid = [model.predict(image[np.newaxis, :, :, :]) for image in tqdm(valid_image)]

    thresholds = np.linspace(0, 1, 50)
    ious = np.array(
        [iou_metric_batch(valid_mask, np.int32(preds_valid > threshold)) for threshold in tqdm(thresholds)])
    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    return threshold_best


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    checkpoint_dir = os.path.join(root_dir, "experiments/salt_detection/checkpoint")
    graph_path = os.path.join(checkpoint_dir, "-845.meta")
    model = Model(checkpoint_dir, graph_path)
    # 计算阈值
    threshold_best = compute_thresholds(model)
    print("threshold_best: {}".format(threshold_best))

    # 读取测试集图片
    root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    test_dir = os.path.join(root_dir, "input/test/images")
    test_list = os.listdir(test_dir)
    test_image = [np.array(load_img(os.path.join(test_dir, img_name), grayscale=True))[:, :, np.newaxis] / 255.0
                  for img_name in tqdm(test_list)]

    preds_test = [model.predict(upsample(image)[np.newaxis, :, :, :]) for image in tqdm(test_image)]

    pred_dict = {
        id[:-4]: rlenc(np.round(downsample(image.reshape((img_size_target, img_size_target)))) > threshold_best) for
        image, id in
        tqdm(zip(preds_test, test_list), total=len(preds_test))}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission.csv')
