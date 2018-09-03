#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : template_trainer.py
# @Time     : 2018/9/3 20:20 
# @Software : PyCharm
from base.base_train import BaseTrain


class TemplateTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(TemplateTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
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
