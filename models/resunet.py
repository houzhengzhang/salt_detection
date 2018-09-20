#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : resunet.py
# @Time     : 2018/9/18 13:02 
# @Software : PyCharm
import tensorflow as tf
from base.base_model import BaseModel
from bunch import Bunch
from models.iou_metric import my_iou_metric
from tensorflow.contrib.slim.nets import resnet_v2, resnet_utils


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


def decoder_conv(input, n_filters):
    with tf.variable_scope("up_decoder"):
        net = tf.layers.conv2d(input, n_filters, kernel_size=(3, 3), activation=None, padding="same", name="conv")
        net = residual_block(net, name="1", num_filters=n_filters)
        net = residual_block(net, name="2", num_filters=n_filters)
        net = tf.nn.relu(net)

        return net


class ResUnet(BaseModel):
    def __init__(self, config):
        super(ResUnet, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_net(self, input, is_training, reuse):
        # format args from config
        args = Bunch(self.config.args)

        inputs = tf.layers.conv2d(input, 3, (1, 1), name="color_space_adjust")
        with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                          args.batch_norm_decay,
                                                          args.batch_norm_epsilon)):
            resnet = getattr(resnet_v2, args.resnet_model)
            _, end_points = resnet(inputs,
                                   args.number_of_classes,
                                   is_training=is_training,
                                   global_pool=False,
                                   output_stride=args.output_stride,
                                   reuse=reuse)
            downname_list = ["resnet_v2_50/conv1", "resnet_v2_50/block1/unit_1/bottleneck_v2",
                             "resnet_v2_50/block1", "resnet_v2_50/block4"]

            n_layers = 4
            down_blocks = []
            for layer_name in downname_list:
                down_blocks.append(end_points[layer_name])

        start_neurons = 256
        feed = down_blocks[-1]
        up_blocks = []
        for i in range(n_layers - 1, 0, -1):
            with tf.variable_scope("unet/up{}".format(i), reuse=reuse):
                up = tf.layers.conv2d_transpose(feed, start_neurons, kernel_size=(3, 3), strides=(2, 2),
                                                padding="same", name="conv2d_transpose_{}".format(i))
                concat = tf.concat([up, down_blocks[i - 1]], axis=-1, name="concat_{}".format(i))
                feed = decoder_conv(concat, start_neurons)
                start_neurons = start_neurons // 2
                up_blocks.append(feed)
                # print("up_blocks:          {}".format(feed))
        with tf.variable_scope("unet/up0", reuse=reuse):
            feed = tf.layers.conv2d_transpose(feed, start_neurons, kernel_size=(3, 3), strides=(2, 2),
                                              padding="same")
            # print(feed)

        with tf.variable_scope("unet/logits", reuse=reuse):
            logits = tf.layers.conv2d(feed, filters=1, kernel_size=(1, 1), padding="same", name="logits",
                                      activation=None)
            # print("logits:    {}".format(logits))

        return logits

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size, name="ori_image")
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size, name="mask_image")
        # 如果断言 pred 为 true 则返回 true_fn() ，否则返回 false_fn()
        # logits_tf = tf.cond(pred=self.is_training,
        #                     true_fn=lambda: self.build_net(self.x, is_training=True, reuse=False),
        #                     false_fn=lambda: self.build_net(self.x, is_training=False, reuse=True))
        self.build_net(self.x, is_training=True, reuse=False)

        variables_to_restore = slim.get_variables_to_restore(exclude=["resnet_v2_50/logits", "optimizer_vars",
                                                                      "unet", "global_step/global_step",
                                                                      "cur_epoch/cur_epoch", "resnet_v2_50/postnorm"
                                                                      ])

        self.restorer = tf.train.Saver(variables_to_restore)
        variables_to_train = slim.get_variables_to_restore(include=["unet"])

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_tf, labels=self.y))

        self.learning_rate = tf.train.exponential_decay(self.config.learning_rate, global_step=self.global_step_tensor,
                                                        decay_steps=3000, decay_rate=0.9,
                                                        staircase=True)
        optimizer_finetune = tf.train.AdamOptimizer(0.00001)
        optimizer_deeplab = tf.train.AdamOptimizer(self.learning_rate)

        grads = tf.gradients(self.cross_entropy, variables_to_restore + variables_to_train)

        grads1 = grads[:len(variables_to_restore)]
        grads2 = grads[len(variables_to_restore):]

        train_op1 = optimizer_finetune.apply_gradients(zip(grads1, variables_to_restore))
        train_op2 = optimizer_deeplab.apply_gradients(zip(grads2, variables_to_train))
        self.train_step = tf.group(train_op1, train_op2)
        # self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy,
        #                                                                       global_step=self.global_step_tensor)
        self.iou_mertic = my_iou_metric(label=self.y, pred=logits_tf)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
