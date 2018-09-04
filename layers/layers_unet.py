#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : layers_unet.py
# @Time     : 2018/9/4 10:29 
# @Software : PyCharm
import tensorflow as tf


def conv_module(input_, n_filters, training, name, pool=True, activation=tf.nn.relu, padding="same", batch_norm=True):
    """{Conv -> BN -> RELU} x 2 -> {Pool, optional}
           reference : https://github.com/kkweon/UNet-in-Tensorflow
       Args:
           input_ (4-D Tensor): (batch_size, H, W, C)
           n_filters (int): depth of output tensor
           training (bool): If True, run in training mode
           name (str): name postfix
           pool (bool): If True, MaxPool2D after last conv layer
           activation: Activaion functions
           padding (str): 'same' or 'valid'
           batch_norm (bool) : If True, use batch-norm
       Returns:
           u_net: output of the Convolution operations
           pool (optional): output of the max pooling operations
       """
    kernel_size = [3, 3]
    net = input_
    with tf.variable_scope("conv_module_{}".format(name)):
        for idx, k_size in enumerate(kernel_size):
            net = tf.layers.conv2d(net, n_filters, (k_size, k_size), activation=None, padding=padding,
                                   name="conv_{}".format(idx + 1))
            if batch_norm:
                net = tf.layers.batch_normalization(net, training=training, renorm=True, name="bn_{}".format(idx + 1))
            net = activation(net, name="relu_{}".format(idx + 1))
        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool")

        return net, pool


def upsample(input_, name, upsacale_factor=(2, 2)):
    h, w, _ = input_.get_shape().as_list()[1:]

    target_h = h * upsacale_factor[0]
    target_w = w * upsacale_factor[1]

    return tf.image.resize_nearest_neighbor(input_, (target_h, target_w), name="upsample_{}".format(name))
