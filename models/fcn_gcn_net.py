#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : fcn_gcn_get.py
# @Time     : 2018/9/10 19:19 
# @Software : PyCharm
import tensorflow as tf
from layers.layers_fcn_gcn import conv_module, global_conv_module, boundary_refine, deconv_module
from base.base_model import BaseModel
from models.iou_metric import my_iou_metric


class FCNGCNnet(BaseModel):
    def __init__(self, config):
        super(FCNGCNnet, self).__init__(config)
        self.init_saver()
        self.build_model()

    def build_net(self, input, is_training=True):
        """Based on https://arxiv.org/abs/1703.02719 but using VGG style base
             Args:
                 input_ (4-D Tensor): (N, H, W, C)
                 is_training (bool): If True, run in training mode
             Returns:
                 output (4-D Tensor): (N, H, W, n)
                     Logits classifying each pixel as either 'car' (1) or 'not car' (0)
             """
        num_classes = 1
        k_gcn = 3
        init_channels = self.config.init_channels  # Number of channels in the first conv layer
        n_layers = self.config.n_layers  # Number of times to downsample/upsample
        batch_norm = self.config.batch_norm  # if True, use batch-norm

        # color-space adjustment
        net = tf.layers.conv2d(input, 3, (1, 1), name="color_space_adjust")
        n = n_layers

        # encoder
        feed = net
        ch = init_channels
        conv_blocks = []
        for i in range(n - 1):
            conv, feed = conv_module(feed, ch, is_training, name=str(i + 1),
                                     batch_norm=batch_norm)
            conv_blocks.append(conv)
            ch *= 2
        last_conv = conv_module(feed, ch, is_training, name=str(n), pool=False,
                                batch_norm=batch_norm)
        conv_blocks.append(last_conv)

        # global convolution network
        ch = init_channels
        global_conv_blocks = []
        for i in range(n):
            print("conv_blocks:{}".format(conv_blocks[i].shape))
            global_conv_blocks.append(
                global_conv_module(conv_blocks[i], 21, is_training,
                                   k=k_gcn, name=str(i + 1)))
            print("global_conv_blocks:{}".format(global_conv_blocks[i].shape))

        # boundary refinement
        br_blocks = []
        for i in range(n):
            br_blocks.append(boundary_refine(global_conv_blocks[i], is_training,
                                             name=str(i + 1), batch_norm=batch_norm))

        # decoder / upsampling
        up_blocks = []
        last_br = br_blocks[-1]

        for i in range(n - 1, 0, -1):
            ch = br_blocks[i-1].get_shape()[3].value
            deconv = deconv_module(last_br, int(ch), name=str(i + 1), stride=2, kernel_size=4)
            up = tf.add(deconv, br_blocks[i - 1])
            last_br = boundary_refine(up, is_training, name='up_' + str(i))
            up_blocks.append(up)

        logits = tf.layers.conv2d(last_br, filters=1, kernel_size=(1, 1), padding="same", name="logits", activation=None)
        return logits

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)

        logits = self.build_net(self.x, is_training=self.is_training)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y))

        self.learning_rate = tf.train.exponential_decay(self.config.learning_rate, global_step=self.global_step_tensor,
                                                        decay_steps=self.config.num_iter_per_epoch, decay_rate=0.9,
                                                        staircase=True)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy,
                                                                              global_step=self.global_step_tensor)
        self.iou_mertic = my_iou_metric(label=self.y, pred=logits)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
