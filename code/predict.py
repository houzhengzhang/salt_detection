#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : predict.py
# @Time     : 2018/8/30 19:41 
# @Software : PyCharm
# Get and resize test images
from keras.models import load_model
from tqdm import tqdm, tnrange
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K

from skimage.transform import resize

from keras.preprocessing.image import img_to_array, load_img

im_width = 128
im_height = 128
im_chan = 1
path_test = '../input/test/'
path_train = '../input/train/'

train_ids = next(os.walk(path_train + "images"))[2]
test_ids = next(os.walk(path_test + "images"))[2]
X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.uint8)

sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = path_test
    img = load_img(path + '/images/' + id_)
    x = img_to_array(img)[:, :, 1]
    sizes_test.append([x.shape[0], x.shape[1]])
    x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
    X_test[n] = x

print('Done!')


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, num_classes=2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            # 获取更新后的 score 值
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
# Predict on train, val and test
model = load_model('model-tgs-salt-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in tnrange(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


pred_dict = {fn[:-4]: RLenc(np.round(preds_test_upsampled[i])) for i, fn in tqdm(enumerate(test_ids))}

sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
