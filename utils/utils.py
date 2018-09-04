#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : utils.py
# @Time     : 2018/9/3 20:55 
# @Software : PyCharm
import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default="None",
        help="The Configuration file")
    args = argparser.parse_args()
    return args
