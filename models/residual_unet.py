#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : residual_unet.py
# @Time     : 2018/9/5 13:43 
# @Software : PyCharm
import tensorflow as tf
from base.base_model import BaseModel
from layers.layers_unet import residual_block, conv_module, encoder_conv, decoder_conv
from models.iou_metric import my_iou_metric


class ResidualUNet(BaseModel):
    def __init__(self, config):
        super(ResidualUNet, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_net(self, input, start_neurons, dropout_ratio=0.5):
        # Number of classes
        num_classes = self.config.num_classes
        # Number of times to downsample/upsample
        init_channels = self.config.init_channels
        # if True, use batch-norm
        batch_norm = self.config.batch_norm
        n_layers = self.config.n_layers

        feed = input
        conv_blocks = []
        # encoder
        for i in range(0, n_layers):
            conv, feed = encoder_conv(feed, start_neurons, dropout_ratio, name="down_{}".format(i + 1))
            conv_blocks.append(conv)
            start_neurons *= 2

        convm = encoder_conv(feed, start_neurons, dropout_ratio=1.0, name="down_{}".format(n_layers + 1), pooling=False)
        conv_blocks.append(convm)

        # decoder
        feed = conv_blocks[-1]
        for i in range(n_layers, 0, -1):
            if i % 2:
                padding = "valid"
            else:
                padding = "same"
            start_neurons /= 2
            up = tf.layers.conv2d_transpose(feed, int(start_neurons), kernel_size=(3, 3), strides=(2, 2),
                                            padding=padding)

            concat = tf.concat([up, conv_blocks[i - 1]], axis=-1, name="concat_".format(i))
            dropout = tf.layers.dropout(concat, dropout_ratio)
            feed = decoder_conv(dropout, start_neurons, name="up_{}".format(i))

        output = tf.layers.dropout(feed, dropout_ratio / 2)
        logits = tf.layers.conv2d(output, filters=1, kernel_size=(1, 1), padding="same", name="logits", activation=None)

        return logits

    def build_model(self):
        self.dropout = tf.placeholder(tf.float32)

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)

        logits = self.build_net(self.x, start_neurons=16, dropout_ratio=self.dropout)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y))

        self.learning_rate = tf.train.exponential_decay(self.config.learning_rate, global_step=self.global_step_tensor,
                                                        decay_steps=self.config.num_iter_per_epoch, decay_rate=0.9,
                                                        staircase=True)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy,
                                                                              global_step=self.global_step_tensor)
        self.iou_mertic = my_iou_metric(label=self.y, pred=logits)

        # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
