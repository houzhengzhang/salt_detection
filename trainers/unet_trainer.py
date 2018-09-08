#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : template_trainer.py
# @Time     : 2018/9/3 20:20 
# @Software : PyCharm
from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class UNetTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(UNetTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)

    def train(self):
        # 开始训练
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        # loop = tqdm(range(self.config.num_iter_per_epoch))
        loop = range(self.config.num_iter_per_epoch)
        losses = []
        accs = []
        iou_mertic = []
        for _ in loop:
            acc, loss, iou = self.train_step()
            losses.append(loss)
            iou_mertic.append(iou)
            accs.append(acc)
        loss = np.mean(losses)
        iou_mertic = np.mean(iou_mertic)
        accs = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            "loss": loss,
            "iou_mertic": iou_mertic,
            "acc": accs
        }
        print("Epoch {}: accuracy: {} iou_mertic: {},loss: {}".format(cur_it, accs, iou_mertic, loss))
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        # 获取 batch 数据
        batch_data = self.sess.run(self.train_data.next_batch())
        feed_dict = {self.model.x: batch_data["images"], self.model.y: batch_data["masks"],
                     self.model.is_training: True}

        _, accuracy, loss, iou_mertic = self.sess.run(
            [self.model.train_step, self.model.accuracy, self.model.cross_entropy, self.model.iou_mertic],
            feed_dict=feed_dict)
        return accuracy, loss, iou_mertic

    def valid_epoch(self):
        iterator = self.valid_data.get_iterator()
        self.sess.run(iterator)
        next_batch = iterator.get_next()
        valid_batches = 20
        for i in range(valid_batches):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def valid_step(self):
        pass
