#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : template_trainer.py
# @Time     : 2018/9/3 20:20 
# @Software : PyCharm
from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class UNetTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(UNetTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            "loss": loss,
            "acc": acc,
        }
        print("Epoch {}: acc {},loss {}".format(cur_it, loss, acc))
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_data = self.sess.run(self.data.next_batch())

        feed_dict = {self.model.x: batch_data["images"], self.model.y: batch_data["masks"], self.model.is_training: True}

        _, loss, acc = self.sess.run([self.model.train_step, self.model.dice_loss, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc
