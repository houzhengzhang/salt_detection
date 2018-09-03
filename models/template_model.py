#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : template_model.py
# @Time     : 2018/9/3 20:15 
# @Software : PyCharm
from base.base_model import BaseModel


class TemplateModel(BaseModel):
    def __init__(self, config):
        super(TemplateModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        raise NotImplementedError

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError
