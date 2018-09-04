#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : losses.py
# @Time     : 2018/9/4 12:27 
# @Software : PyCharm
import tensorflow as tf


# Precison = TP/(TP + FP)
# Specificity(特异性) = TN /(TN + FP)
# Recall = Sensitivity(灵敏性) = TP /(TP + FN)
# 调和平均
# 1/F1 = 1/Precison + 1/Recall
# F1 = 2 * TP /(2 * TP + FP + FN) = 2 |X ∩ Y| / (|X| + |Y|)

def dice_loss(y_true, y_pred, axis=None, smooth=0.001):
    if axis is None:
        axis = [1, 2]

    y_true_f = tf.cast(y_true, dtype=tf.float32)
    y_pred_f = tf.cast(y_pred, dtype=tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=axis)
    coefficient = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=axis)
                                                  + tf.reduce_sum(y_pred_f, axis=axis) + smooth)
    loss = 1. - tf.reduce_mean(coefficient)
    return loss


def pixel_wise_loss(pixel_logits, gt_pixels, pixel_weights=None):
    """Calculates pixel-wise softmax cross entropy loss
    Args:
        pixel_logits (4-D Tensor): (N, H, W, 2)
        gt_pixels (3-D Tensor): Image masks of shape (N, H, W, 2)
        pixel_weights (3-D Tensor) : (N, H, W) Weights for each pixel
    Returns:
        scalar loss : softmax cross-entropy
    """
    logits = tf.reshape(pixel_logits, [-1, 2])
    labels = tf.reshape(gt_pixels, [-1, 2])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    if pixel_weights is None:
        return tf.reduce_mean(loss)
    else:
        weights = tf.reshape(pixel_weights, [-1])
        return tf.reduce_sum(loss * weights) / tf.reduce_sum(weights)


def mask_prediction(pixel_logits):
    """
    Args:
        pixel_logits (4-D Tensor): (N, H, W, 2)
    Returns:
        Predicted pixel-wise probabilities (3-D Tensor): (N, H, W)
        Predicted mask (3-D Tensor): (N, H, W)
    """
    probs = tf.nn.softmax(pixel_logits)
    n, h, w, _ = probs.get_shape()
    masks = tf.reshape(probs, [-1, 2])
    masks = tf.argmax(masks, axis=1)
    masks = tf.reshape(masks, [n.value, h.value, w.value])
    probs = tf.slice(probs, [0, 0, 0, 1], [-1, -1, -1, 1])
    probs = tf.squeeze(probs, axis=-1)
    return probs, masks
