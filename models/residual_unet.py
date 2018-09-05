#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : residual_unet.py
# @Time     : 2018/9/5 13:43 
# @Software : PyCharm
import tensorflow as tf
from base.base_model import BaseModel
from layers.layers_unet import residual_block, conv_module


class ResidualUNet(BaseModel):
    def __init__(self, config):
        super(ResidualUNet, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_net(self, input, start_neurons, dropout_ratio=0.5, is_training=False):
        # Number of classes
        num_classes = self.config.num_classes
        # Number of times to downsample/upsample
        init_channels = self.config.init_channels
        # if True, use batch-norm
        batch_norm = self.config.batch_norm
        # 101 -> 50
        conv1 = tf.layers.conv2d(input, filters=start_neurons * 1, kernel_size=(3, 3), activation=None,
                                 padding="same", name="conv1")

        conv1 = residual_block(conv1, name="1", num_filters=start_neurons * 1)
        conv1 = residual_block(conv1, name="2", num_filters=start_neurons * 1)
        conv1 = tf.nn.relu(conv1)

        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), name="pool")
        pool1 = tf.layers.dropout(pool1, dropout_ratio / 2)

        # 50 -> 25
        conv2 = tf.layers.conv2d(pool1, filters=start_neurons * 2, kernel_size=(3, 3), activation=None,
                                 padding="same")
        conv2 = residual_block(conv2, name="3", num_filters=start_neurons * 2)
        conv2 = residual_block(conv2, name="4", num_filters=start_neurons * 2)
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name="pool")
        pool2 = tf.layers.dropout(pool2, dropout_ratio)

        # 25 -> 12
        conv3 = tf.layers.conv2d(pool2, filters=start_neurons * 4, kernel_size=(3, 3), activation=None,
                                 padding="same")
        conv3 = residual_block(conv3, name="5", num_filters=start_neurons * 4)
        conv3 = residual_block(conv3, name="6", num_filters=start_neurons * 4)
        conv3 = tf.nn.relu(conv3)
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), name="pool")
        pool3 = tf.layers.dropout(pool3, dropout_ratio)

        # 12 -> 6
        conv4 = tf.layers.conv2d(pool3, filters=start_neurons * 8, kernel_size=(3, 3), activation=None,
                                 padding="same")
        conv4 = residual_block(conv4, name="7", num_filters=start_neurons * 8)
        conv4 = residual_block(conv4, name="8", num_filters=start_neurons * 8)
        conv4 = tf.nn.relu(conv4)
        pool4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), name="pool")
        pool4 = tf.layers.dropout(pool4, dropout_ratio)

        # Middle
        convm = tf.layers.conv2d(pool4, start_neurons * 16, kernel_size=(3, 3), activation=None,
                                 padding="same")
        convm = residual_block(convm, name="9", num_filters=start_neurons * 16)
        convm = residual_block(convm, name="10", num_filters=start_neurons * 16)
        convm = tf.nn.relu(convm)

        # 6 -> 12
        dconv4 = tf.layers.conv2d_transpose(convm, start_neurons * 8, kernel_size=(3, 3), strides=(2, 2),
                                            padding="same")

        uconv4 = tf.concat([dconv4, conv4], axis=-1)
        uconv4 = tf.layers.dropout(uconv4, dropout_ratio)

        uconv4 = tf.layers.conv2d(uconv4, start_neurons * 8, kernel_size=(3, 3), activation=None, padding="same")
        uconv4 = residual_block(uconv4, name="11", num_filters=start_neurons * 8)
        uconv4 = residual_block(uconv4, name="12", num_filters=start_neurons * 8)
        uconv4 = tf.nn.relu(uconv4)

        # 12 -> 25
        uconv3 = tf.layers.conv2d_transpose(uconv4, start_neurons * 4, kernel_size=(3, 3), strides=(2, 2),
                                            padding="valid")
        uconv3 = tf.concat([uconv3, conv3], axis=-1)
        uconv3 = tf.layers.dropout(uconv3, dropout_ratio)

        uconv3 = tf.layers.conv2d(uconv3, start_neurons * 4, kernel_size=(3, 3), activation=None, padding="same")
        uconv3 = residual_block(uconv3, name="13", num_filters=start_neurons * 4)
        uconv3 = residual_block(uconv3, name="14", num_filters=start_neurons * 4)
        uconv3 = tf.nn.relu(uconv3)

        # 25 -> 50
        uconv2 = tf.layers.conv2d_transpose(uconv3, start_neurons * 2, kernel_size=(3, 3), strides=(2, 2),
                                            padding="same")
        uconv2 = tf.concat([uconv2, conv2], axis=-1)
        uconv2 = tf.layers.dropout(uconv2, dropout_ratio)

        uconv2 = tf.layers.conv2d(uconv2, start_neurons * 2, kernel_size=(3, 3), activation=None, padding="same")
        uconv2 = residual_block(uconv2, name="15", num_filters=start_neurons * 2)
        uconv2 = residual_block(uconv2, name="16", num_filters=start_neurons * 2)
        uconv2 = tf.nn.relu(uconv2)

        # 50 -> 101
        uconv1 = tf.layers.conv2d_transpose(uconv2, start_neurons * 1, kernel_size=(3, 3), strides=(2, 2),
                                            padding="valid")
        uconv1 = tf.concat([uconv1, conv1], axis=-1)
        uconv1 = tf.layers.dropout(uconv1, dropout_ratio)

        uconv1 = tf.layers.conv2d(uconv1, start_neurons * 1, kernel_size=(3, 3), activation=None, padding="same")
        uconv1 = residual_block(uconv1, name="17", num_filters=start_neurons * 1)
        uconv1 = residual_block(uconv1, name="18", num_filters=start_neurons * 1)
        uconv1 = tf.nn.relu(uconv1)

        uconv1 = tf.layers.dropout(uconv1, dropout_ratio / 2)
        output_layer = tf.layers.conv2d(uconv1, 1, kernel_size=(1, 1), padding="same")

        return output_layer

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)

        logits = self.build_net(self.x, start_neurons=16, dropout_ratio=0.5, is_training=self.is_training)

        with tf.name_scope("loss"):
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)

        self.learning_rate = tf.train.exponential_decay(self.config.learning_rate, global_step=self.global_step_tensor,
                                                        decay_steps=self.config.num_iter_per_epoch, decay_rate=0.9,
                                                        staircase=True)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                              global_step=self.global_step_tensor)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
