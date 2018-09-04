#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : main.py
# @Time     : 2018/9/3 21:06 
# @Software : PyCharm
import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from data_loader.data_generator import DataGenerator
from models.u_net import UNet
from trainers.unet_trainer import UNetTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except FileNotFoundError as e:
        print(e)
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    # 指定使用显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.90  # 占用GPU90%的显存

    sess = tf.Session(config=tf_config)

    # create data generator
    data = DataGenerator(config)

    # create an model
    model = UNet(config)

    # create tensorboard logger
    logger = Logger(sess, config)

    # create trainer and pass all the previous compoents to it
    trainer = UNetTrainer(sess, model, data, config, logger)

    # load model if exists
    model.load(sess)

    # train model
    trainer.train()


if __name__ == '__main__':
    main()
