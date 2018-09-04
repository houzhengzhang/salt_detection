#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : template_model.py
# @Time     : 2018/9/3 20:15 
# @Software : PyCharm
from base.base_model import BaseModel
from models.losses import dice_loss
import tensorflow as tf
from layers.layers_unet import conv_module, upsample


class UNet(BaseModel):
    def __init__(self, config):
        super(UNet, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_net(self, input, is_training=False):
        """
               Args:
                   input (4-D Tensor): (N, H, W, C)
                   is_training (bool): If True, run in training mode
               Returns:
                   output (4-D Tensor): (N, H, W, n)
                       Logits classifying each pixel as either 'car' (1) or 'not car' (0)
               """
        # Number of classes
        num_classes = self.config.num_classes
        # Number of times to downsample/upsample
        n_layers = self.config.n_layers
        # Number of channels in the first conv layer
        init_channels = self.config.init_channels
        # if True, use batch-norm
        batch_norm = self.config.batch_norm

        # color-space adjustment
        net = tf.layers.conv2d(input, 3, (1, 1), name="color_space_adjust")

        # encoder
        feed = net
        ch = init_channels
        conv_blocks = []
        for i in range(n_layers):
            # net pool
            conv, feed = conv_module(feed, ch, is_training, name="down_{}".format(i + 1), batch_norm=batch_norm)
            conv_blocks.append(conv)

            ch *= 2
        last_conv = conv_module(feed, ch, is_training, name="down_{}".format(n_layers + 1), pool=False,
                                batch_norm=batch_norm)
        conv_blocks.append(last_conv)

        # decoder / upsampling
        feed = conv_blocks[-1]
        for i in range(n_layers, 0, -1):
            ch /= 2
            up = upsample(feed, name=str(i + 1))
            concat = tf.concat([up, conv_blocks[i - 1]], axis=-1, name="concat_{}".format(i))
            feed = conv_module(concat, ch, is_training, name="up_{}".format(i), batch_norm=batch_norm, pool=False)

        logits = tf.layers.conv2d(feed, num_classes, (1, 1), name="logits", activation=None, padding="same")

        return logits

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)

        logits = self.build_net(self.x, self.is_training)

        with tf.name_scope("loss"):
            self.dice_loss = dice_loss(y_true=self.y, y_pred=logits)
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.dice_loss,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
