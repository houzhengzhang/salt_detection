#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : write_tfrecord.py
# @Time     : 2018/9/7 22:47 
# @Software : PyCharm
import numpy as np
import pandas as pd
from skimage.transform import resize
from tqdm import tqdm
import tensorflow as tf
from keras.preprocessing.image import load_img
from collections import defaultdict
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from utils.tf_utils import _bytes_feature

img_size_ori = 101
img_size_target = 128


def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


# 根据覆盖面积分为11个类
def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


def read_data(traincsv_dir, depthcsv_dir, train_dir):
    train_df = pd.read_csv(os.path.join(traincsv_dir, "train.csv"), index_col="id", usecols=[0])
    # 22000
    depths_df = pd.read_csv(os.path.join(depthcsv_dir, "depths.csv"), index_col="id")
    # id z 4000
    train_df = train_df.join(depths_df)

    train_df["images"] = [np.array(load_img(os.path.join(train_dir, "images/{}.png".format(idx)),
                                            grayscale=True)) / 255 for idx in tqdm(train_df.index)]
    train_df["masks"] = [np.array(load_img(os.path.join(train_dir, "masks/{}.png".format(idx)),
                                           grayscale=True)) / 255 for idx in tqdm(train_df.index)]

    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_target, 2)

    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    del train_df["z"]
    del train_df["coverage"]

    train_dict = defaultdict(list)
    for images, masks, coverage_class in train_df.values:
        train_dict[coverage_class].append((images, masks))

    return train_dict


def train_test_split(data_dict, split_scale=0.9):
    train_data = []
    valid_data = []
    for key, values in data_dict.items():
        patrition = int(len(values) * split_scale)
        train_data.append(values[:patrition])
        valid_data.append(values[patrition:])

    train_images = []
    train_masks = []
    for image_tuple in train_data:
        for images, masks in image_tuple:
            train_images.append(images)
            train_masks.append(masks)

    valid_images = []
    valid_masks = []
    for image_tuple in valid_data:
        for images, masks in image_tuple:
            valid_images.append(images)
            valid_masks.append(masks)

    # 数据集扩增
    # 左右翻转
    train_images_lr = [np.fliplr(x) for x in train_images]
    # 上下翻转
    train_images_ud = [np.flipud(x) for x in train_images]
    train_images = np.append(train_images, train_images_lr, axis=0)
    train_images = np.append(train_images, train_images_ud, axis=0)

    train_masks_lr = [np.fliplr(x) for x in train_masks]
    train_masks_ud = [np.flipud(x) for x in train_masks]
    train_masks = np.append(train_masks, train_masks_lr, axis=0)
    train_masks = np.append(train_masks, train_masks_ud, axis=0)

    valid_images = np.append(valid_images, [np.fliplr(x) for x in valid_images], axis=0)
    valid_masks = np.append(valid_masks, [np.fliplr(x) for x in valid_masks], axis=0)

    return train_images, train_masks, valid_images, valid_masks


def write_tfrecord(tfrecord_path, x_train, y_train):
    with tf.python_io.TFRecordWriter(tfrecord_path) as tfrecord_writer:
        for img, mask in tqdm(zip(x_train, y_train), total=len(x_train)):
            img_data = upsample(img).tobytes()
            mask_data = upsample(mask).tobytes()
            # img_data = img.tobytes()
            # mask_data = mask.tobytes()
            # create features
            feature = {'images': _bytes_feature(img_data),
                       'masks': _bytes_feature(mask_data)}
            # create example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # serialize protocol buffer to string
            tfrecord_writer.write(example.SerializeToString())


if __name__ == '__main__':
    root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    traincsv_dir = os.path.join(root_dir, "input")
    depthcsv_dir = os.path.join(root_dir, "input")
    train_dir = os.path.join(root_dir, "input/train")

    data_dict = read_data(traincsv_dir, depthcsv_dir, train_dir)
    train_images, train_masks, valid_images, valid_masks = train_test_split(data_dict, split_scale=0.9)

    write_tfrecord(os.path.join(root_dir, "input", "train.tfrecords"), train_images, train_masks)
    write_tfrecord(os.path.join(root_dir, "input", "valid.tfrecords"), valid_images, valid_masks)
