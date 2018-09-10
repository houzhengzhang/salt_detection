#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : layers_unet.py
# @Time     : 2018/9/4 10:29 
# @Software : PyCharm
import tensorflow as tf

slim = tf.contrib.slim


def convolution_block(x, num_filters, name, kernel_size=(3, 3), activation=True):
    with tf.variable_scope("conv_{}".format(name)):
        net = tf.layers.conv2d(x, num_filters, kernel_size, padding="same")
        net = tf.layers.batch_normalization(net, renorm=True)
        if activation:
            net = tf.nn.relu(net, name="relu")
    return net


def residual_block(block_input, name, num_filters=16):
    with tf.variable_scope("residual_block_{}".format(name)):
        net = tf.nn.relu(block_input, name="relu")
        net = tf.layers.batch_normalization(net, renorm=True)
        net = convolution_block(net, num_filters, name="1", kernel_size=(3, 3))
        net = convolution_block(net, num_filters, name="2", kernel_size=(3, 3))
        net = convolution_block(net, num_filters, name="3", kernel_size=(3, 3), activation=False)
        add = tf.add(net, block_input)
    return add


def encoder_conv(input, n_filters, dropout_ratio, name, pooling=True):
    with tf.variable_scope("unet_model_{}".format(name)):
        net = tf.layers.conv2d(input, filters=n_filters, kernel_size=(3, 3), activation=None,
                               padding="same", name="conv1")

        net = residual_block(net, name="1", num_filters=n_filters)
        net = residual_block(net, name="2", num_filters=n_filters)
        net = tf.nn.relu(net)

        if pooling is False:
            return net
        pool = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(2, 2), name="pool")
        pool = tf.layers.dropout(pool, dropout_ratio)

        return net, pool


def decoder_conv(input, n_filters, name):
    with tf.variable_scope("unet_model_{}".format(name)):
        net = tf.layers.conv2d(input, n_filters, kernel_size=(3, 3), activation=None, padding="same", name="conv")
        net = residual_block(net, name="1", num_filters=n_filters)
        net = residual_block(net, name="2", num_filters=n_filters)
        net = tf.nn.relu(net)

        return net
