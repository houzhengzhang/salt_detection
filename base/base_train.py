#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : base_train.py
# @Time     : 2018/9/3 19:48 
# @Software : PyCharm
import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, train_data, valid_data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.train_data = train_data
        self.valid_data = valid_data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        raise NotImplementedError

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def valid(self):
        raise NotImplemented
