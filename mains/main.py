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
from data_loader.valid_data_generator import ValidDataGenerator
from models.residual_unet import ResidualUNet
from models.fcn_gcn_net import FCNGCNnet
from models.resunet import ResUnet
from trainers.unet_trainer import UNetTrainer
from trainers.fcn_trainer import FCNTrainer
from trainers.resunet_trainer import ResunetTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def count1():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)


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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.85  # 占用GPU90%的显存

    sess = tf.Session(config=tf_config)

    # create data generator
    train_data = DataGenerator(config)
    valid_data = ValidDataGenerator(config)

    # create tensorboard logger
    logger = Logger(sess, config)

    # create an model
    if args.model == "unet":
        model = ResidualUNet(config)
        # create trainer and pass all the previous compoents to it
        trainer = UNetTrainer(sess, model, train_data, valid_data, config, logger)
    elif args.model == "fcn":
        model = FCNGCNnet(config)
        trainer = FCNTrainer(sess, model, train_data, valid_data, config, logger)
    elif args.model == "resunet":
        model = ResUnet(config)
        trainer = ResunetTrainer(sess, model, train_data, valid_data, config, logger)
        model.restorer.restore(sess, "../resnet/resnet_v2_50.ckpt")
        print("load weights done!")
    # load model if exists
    model.load(sess)

    # train model
    trainer.train()

    # usage
    # The best I get right now is 2.0 * dice_loss + 1.0 * binary_crossentropy.
    # 我们使用Resnet-34编码器在U-Net上从0.820增加到0.830。
    # pid 6206
    # nohup python3 -u  main.py --c="../configs/config.json"  > logs.out 2>&1 &
    # python3 main.py --c="./configs/config.json"


if __name__ == '__main__':
    main()
