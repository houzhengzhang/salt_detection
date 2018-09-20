#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : resunet_trainer.py
# @Time     : 2018/9/18 12:20 
# @Software : PyCharm
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from base.base_train import BaseTrain


class ResunetTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(ResunetTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)
        self.valid_losses = np.inf

    def train(self):
        # 开始训练
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        # loop = tqdm(range(self.config.num_iter_per_epoch))
        start_time = time.time()
        loop = range(self.config.num_iter_per_epoch)
        losses = []
        iou_mertic = []
        for _ in loop:
            try:
                loss, iou = self.train_step()
            except tf.errors.OutOfRangeError:
                print("train finished!")
                exit(0)
            losses.append(loss)
            iou_mertic.append(iou)

        loss = np.mean(losses)
        iou_mertic = np.mean(iou_mertic)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            "loss": loss,
            "iou_mertic": iou_mertic,
            "learning_rate": self.sess.run(self.model.learning_rate)
        }

        valid_loss, valid_iou = self.valid()
        print("Epoch {}: iou_mertic: {},loss: {} \
                 valid_iou: {} valid_loss: {} time: {}s".format(cur_it, round(float(iou_mertic), 4),
                                                                round(float(loss), 4),
                                                                round(float(valid_iou), 4), round(float(valid_loss), 4),
                                                                round(time.time() - start_time, 2)))

        self.logger.summarize(cur_it, summarizer="train", summaries_dict=summaries_dict)
        valid_summaries_dict = {
            "loss": valid_loss,
            "iou_mertic": valid_iou,
        }
        self.logger.summarize(cur_it, summarizer="test", summaries_dict=valid_summaries_dict)
        # if valid_loss < self.valid_losses:
        #     print("valid loss improved from {} to {}".format(self.valid_losses, valid_loss))
        #     self.valid_losses = valid_loss
        self.model.save(self.sess)

    def train_step(self):
        # 获取 batch 数据
        batch_data = self.sess.run(self.train_data.next_batch())
        feed_dict = {self.model.x: batch_data["images"], self.model.y: batch_data["masks"],
                     self.model.is_training: True}

        _, loss, iou_mertic = self.sess.run(
            [self.model.train_step, self.model.cross_entropy, self.model.iou_mertic], feed_dict=feed_dict)

        return loss, iou_mertic

    def valid(self):
        iterator = self.valid_data.get_iterator()
        next_batch = iterator.get_next()
        # 初始化迭代器
        self.sess.run(iterator.initializer)
        losses = []
        iou_mertic = []

        while True:
            try:
                batch_data = self.sess.run(next_batch)
                feed_dict = {self.model.x: batch_data["images"], self.model.y: batch_data["masks"],
                             self.model.is_training: False}
                loss, iou = self.sess.run(
                    [self.model.cross_entropy, self.model.iou_mertic],
                    feed_dict=feed_dict)
                losses.append(loss)
                iou_mertic.append(iou)
            except tf.errors.OutOfRangeError:
                # 验证集测试完成
                pass
            finally:
                return np.mean(losses), np.mean(iou_mertic)
