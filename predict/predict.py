#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : predict.py
# @Time     : 2018/9/7 14:16 
# @Software : PyCharm
import tensorflow as tf
import os
from keras.preprocessing.image import load_img
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
from skimage.transform import resize


class Model:
    def __init__(self, checkpoint_name, graph_name):
        self.checkpoint_name = checkpoint_name
        self.graph_name = graph_name
        self.load_weight()
        print("weight load done!")

    def load_weight(self):
        checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_name)
        # 获取默认图
        self.graph = tf.get_default_graph()
        self.sess = tf.Session()

        self.saver = tf.train.import_meta_graph(self.graph_name)
        self.saver.restore(self.sess, checkpoint_file)

        self.input_op = self.graph.get_operation_by_name("Placeholder_1").outputs[0]
        self.logits = self.graph.get_operation_by_name("logits/Conv2D").outputs[0]

    def predict(self, input):
        logits = self.sess.run(self.logits, feed_dict={self.input_op: input})
        return logits.reshape((-1, 101, 101))


def mask_img(input):
    result = []
    masks = input.reshape((101 * 101))

    for pixe in masks:
        if pixe:
            result.append(1)
        else:
            result.append(0)

    return np.array(result).reshape((101, 101))


img_size_ori = 101
img_size_target = 101


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


def predict_result(model, x_test, img_size_target):  # predict both orginal and reflect x
    x_test_reflect = np.array([np.fliplr(x) for x in x_test])
    preds_test1 = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test2 = np.array([np.fliplr(x) for x in preds_test2_refect])
    preds_avg = (preds_test1 + preds_test2) / 2
    return preds_avg


if __name__ == '__main__':
    checkpoint_name = "D:\\PythonProject\\salt_detection\\experiments\\salt_detection\\checkpoint"
    graph_name = "D:\\PythonProject\\salt_detection\\experiments\\salt_detection\\checkpoint\\-10875.meta"
    model = Model(checkpoint_name, graph_name)

    root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    test_dir = os.path.join(root_dir, "input/test/images")
    test_list = os.listdir(test_dir)[:50]
    test_image = [np.array(load_img(os.path.join(test_dir, img_name), grayscale=True)) / 255.0 for img_name in tqdm(test_list)]

    preds_valid = predict_result(model, test_image, 101)
    preds_valid2 = np.array([downsample(x) for x in preds_valid])

    y_valid2 = np.array([downsample(x) for x in y_valid])

    ## Scoring for last model
    thresholds = np.linspace(0.3, 0.7, 31)
    ious = np.array(
        [iou_metric_batch(y_valid2, np.int32(preds_valid2 > threshold)) for threshold in tqdm(thresholds)])

    one_image = test_image[0][np.newaxis, :, :, np.newaxis] / 255.0
    predict_img = model.predict(one_image)
    print(predict_img[0].shape)
    # print(np.round(predict_img[0]))
    masks = mask_img(np.round(predict_img[0]) > 0.3)

    photo = Image.fromarray(masks, mode="1")
    photo.save("test2.jpg")
