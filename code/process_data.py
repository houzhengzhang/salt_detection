#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : process_data.py
# @Time     : 2018/8/30 21:26 
# @Software : PyCharm
import numpy as np
import pandas as pd
import six

from sklearn.model_selection import train_test_split

from skimage.transform import resize

from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from tqdm import tqdm
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.regularizers import l2

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

from keras.preprocessing.image import load_img

img_size_ori = 101
img_size_target = 128


def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    # return img[:img_size_ori, :img_size_ori]


train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
# 22000
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
# id z 4000
train_df = train_df.join(depths_df)
# 18000
test_df = depths_df[~depths_df.index.isin(train_df.index)]

train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx),
                                        grayscale=True)) / 255 for idx in tqdm(train_df.index)]

train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx),
                                       grayscale=True)) / 255 for idx in tqdm(train_df.index)]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)


# 根据覆盖面积分为11个类
def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state=1337)

###################################
#       Build U-Net Model         #
###################################


def handle_block_names(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name


def Upsample2D_block(filters, stage, kernel_size=(3, 3), upsample_rate=(2, 2),
                     batchnorm=False, skip=None):
    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = Conv2D(filters, kernel_size, padding='same', name=conv_name + '1')(x)
        if batchnorm:
            x = BatchNormalization(name=bn_name + '1')(x)
        x = Activation('relu', name=relu_name + '1')(x)

        x = Conv2D(filters, kernel_size, padding='same', name=conv_name + '2')(x)
        if batchnorm:
            x = BatchNormalization(name=bn_name + '2')(x)
        x = Activation('relu', name=relu_name + '2')(x)

        return x

    return layer


def Transpose2D_block(filters, stage, kernel_size=(3, 3), upsample_rate=(2, 2),
                      transpose_kernel_size=(4, 4), batchnorm=False, skip=None):
    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name)(input_tensor)
        if batchnorm:
            x = BatchNormalization(name=bn_name + '1')(x)
        x = Activation('relu', name=relu_name + '1')(x)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = Conv2D(filters, kernel_size, padding='same', name=conv_name + '2')(x)
        if batchnorm:
            x = BatchNormalization(name=bn_name + '2')(x)
        x = Activation('relu', name=relu_name + '2')(x)

        return x

    return layer


def get_layer_number(model, layer_name):
    """
    Help find layer in Keras model by name
    Args:
        model: Keras `Model`
        layer_name: str, name of layer
    Returns:
        index of layer
    Raises:
        ValueError: if model does not contains layer with such name
    """
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    raise ValueError('No layer with name {} in  model {}.'.format(layer_name, model.name))


def build_unet(backbone, classes, last_block_filters, skip_layers,
               n_upsample_blocks=5, upsample_rates=(2, 2, 2, 2, 2),
               block_type='upsampling', activation='sigmoid',
               **kwargs):
    input = backbone.input
    x = backbone.output

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # convert layer names to indices
    skip_layers = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                    for l in skip_layers])
    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        if i < len(skip_layers):
            #             print(backbone.layers[skip_layers[i]])
            #             print(backbone.layers[skip_layers[i]].output)
            skip = backbone.layers[skip_layers[i]].output
        else:
            skip = None

        up_size = (upsample_rates[i], upsample_rates[i])
        filters = last_block_filters * 2 ** (n_upsample_blocks - (i + 1))

        x = up_block(filters, i, upsample_rate=up_size, skip=skip, **kwargs)(x)

    if classes < 2:
        activation = 'sigmoid'

    x = Conv2D(classes, (3, 3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model


# https://github.com/raghakot/keras-resnet/blob/master/resnet.py
def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    """

    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """

    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, block_fn, repetitions, input_tensor):
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(img_input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        model = Model(inputs=img_input, outputs=block)
        return model

    @staticmethod
    def build_resnet_34(input_shape, input_tensor):
        return ResnetBuilder.build(input_shape, basic_block, [3, 4, 6, 3], input_tensor)


def UResNet34(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='upsampling',
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):
    backbone = ResnetBuilder.build_resnet_34(input_shape=input_shape, input_tensor=input_tensor)

    skip_connections = list([97, 54, 25])  # for resnet 34
    model = build_unet(backbone, classes, decoder_filters,
                       skip_connections, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-resnet34'

    return model


from keras.losses import binary_crossentropy
from keras import backend as K


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) +
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
        y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss


if __name__ == '__main__':
    import os
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.90  # 占用GPU90%的显存
    K.set_session(tf.Session(config=config))

    # 创建模型
    model = UResNet34(input_shape=(1, img_size_target, img_size_target))
    model.summary()
    exit()
    # 创建优化器
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=bce_dice_loss, optimizer=opt, metrics=["accuracy"])

    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint("./salt.h5", save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=4, min_lr=0.00001, verbose=1)

    epochs = 200
    batch_size = 32

    history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid],
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint, reduce_lr], shuffle=True, verbose=1)


# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            (None, 128, 128, 1)  0
# __________________________________________________________________________________________________
# conv2d_1 (Conv2D)               (None, 64, 64, 64)   3200        input_1[0][0]
# __________________________________________________________________________________________________
# batch_normalization_1 (BatchNor (None, 64, 64, 64)   256         conv2d_1[0][0]
# __________________________________________________________________________________________________
# activation_1 (Activation)       (None, 64, 64, 64)   0           batch_normalization_1[0][0]
# __________________________________________________________________________________________________
# max_pooling2d_1 (MaxPooling2D)  (None, 32, 32, 64)   0           activation_1[0][0]
# __________________________________________________________________________________________________
# conv2d_2 (Conv2D)               (None, 32, 32, 64)   36928       max_pooling2d_1[0][0]
# __________________________________________________________________________________________________
# batch_normalization_2 (BatchNor (None, 32, 32, 64)   256         conv2d_2[0][0]
# __________________________________________________________________________________________________
# activation_2 (Activation)       (None, 32, 32, 64)   0           batch_normalization_2[0][0]
# __________________________________________________________________________________________________
# conv2d_3 (Conv2D)               (None, 32, 32, 64)   36928       activation_2[0][0]
# __________________________________________________________________________________________________
# add_1 (Add)                     (None, 32, 32, 64)   0           max_pooling2d_1[0][0]
#                                                                  conv2d_3[0][0]
# __________________________________________________________________________________________________
# batch_normalization_3 (BatchNor (None, 32, 32, 64)   256         add_1[0][0]
# __________________________________________________________________________________________________
# activation_3 (Activation)       (None, 32, 32, 64)   0           batch_normalization_3[0][0]
# __________________________________________________________________________________________________
# conv2d_4 (Conv2D)               (None, 32, 32, 64)   36928       activation_3[0][0]
# __________________________________________________________________________________________________
# batch_normalization_4 (BatchNor (None, 32, 32, 64)   256         conv2d_4[0][0]
# __________________________________________________________________________________________________
# activation_4 (Activation)       (None, 32, 32, 64)   0           batch_normalization_4[0][0]
# __________________________________________________________________________________________________
# conv2d_5 (Conv2D)               (None, 32, 32, 64)   36928       activation_4[0][0]
# __________________________________________________________________________________________________
# add_2 (Add)                     (None, 32, 32, 64)   0           add_1[0][0]
#                                                                  conv2d_5[0][0]
# __________________________________________________________________________________________________
# batch_normalization_5 (BatchNor (None, 32, 32, 64)   256         add_2[0][0]
# __________________________________________________________________________________________________
# activation_5 (Activation)       (None, 32, 32, 64)   0           batch_normalization_5[0][0]
# __________________________________________________________________________________________________
# conv2d_6 (Conv2D)               (None, 32, 32, 64)   36928       activation_5[0][0]
# __________________________________________________________________________________________________
# batch_normalization_6 (BatchNor (None, 32, 32, 64)   256         conv2d_6[0][0]
# __________________________________________________________________________________________________
# activation_6 (Activation)       (None, 32, 32, 64)   0           batch_normalization_6[0][0]
# __________________________________________________________________________________________________
# conv2d_7 (Conv2D)               (None, 32, 32, 64)   36928       activation_6[0][0]
# __________________________________________________________________________________________________
# add_3 (Add)                     (None, 32, 32, 64)   0           add_2[0][0]
#                                                                  conv2d_7[0][0]
# __________________________________________________________________________________________________
# batch_normalization_7 (BatchNor (None, 32, 32, 64)   256         add_3[0][0]
# __________________________________________________________________________________________________
# activation_7 (Activation)       (None, 32, 32, 64)   0           batch_normalization_7[0][0]
# __________________________________________________________________________________________________
# conv2d_8 (Conv2D)               (None, 16, 16, 128)  73856       activation_7[0][0]
# __________________________________________________________________________________________________
# batch_normalization_8 (BatchNor (None, 16, 16, 128)  512         conv2d_8[0][0]
# __________________________________________________________________________________________________
# activation_8 (Activation)       (None, 16, 16, 128)  0           batch_normalization_8[0][0]
# __________________________________________________________________________________________________
# conv2d_10 (Conv2D)              (None, 16, 16, 128)  8320        add_3[0][0]
# __________________________________________________________________________________________________
# conv2d_9 (Conv2D)               (None, 16, 16, 128)  147584      activation_8[0][0]
# __________________________________________________________________________________________________
# add_4 (Add)                     (None, 16, 16, 128)  0           conv2d_10[0][0]
#                                                                  conv2d_9[0][0]
# __________________________________________________________________________________________________
# batch_normalization_9 (BatchNor (None, 16, 16, 128)  512         add_4[0][0]
# __________________________________________________________________________________________________
# activation_9 (Activation)       (None, 16, 16, 128)  0           batch_normalization_9[0][0]
# __________________________________________________________________________________________________
# conv2d_11 (Conv2D)              (None, 16, 16, 128)  147584      activation_9[0][0]
# __________________________________________________________________________________________________
# batch_normalization_10 (BatchNo (None, 16, 16, 128)  512         conv2d_11[0][0]
# __________________________________________________________________________________________________
# activation_10 (Activation)      (None, 16, 16, 128)  0           batch_normalization_10[0][0]
# __________________________________________________________________________________________________
# conv2d_12 (Conv2D)              (None, 16, 16, 128)  147584      activation_10[0][0]
# __________________________________________________________________________________________________
# add_5 (Add)                     (None, 16, 16, 128)  0           add_4[0][0]
#                                                                  conv2d_12[0][0]
# __________________________________________________________________________________________________
# batch_normalization_11 (BatchNo (None, 16, 16, 128)  512         add_5[0][0]
# __________________________________________________________________________________________________
# activation_11 (Activation)      (None, 16, 16, 128)  0           batch_normalization_11[0][0]
# __________________________________________________________________________________________________
# conv2d_13 (Conv2D)              (None, 16, 16, 128)  147584      activation_11[0][0]
# __________________________________________________________________________________________________
# batch_normalization_12 (BatchNo (None, 16, 16, 128)  512         conv2d_13[0][0]
# __________________________________________________________________________________________________
# activation_12 (Activation)      (None, 16, 16, 128)  0           batch_normalization_12[0][0]
# __________________________________________________________________________________________________
# conv2d_14 (Conv2D)              (None, 16, 16, 128)  147584      activation_12[0][0]
# __________________________________________________________________________________________________
# add_6 (Add)                     (None, 16, 16, 128)  0           add_5[0][0]
#                                                                  conv2d_14[0][0]
# __________________________________________________________________________________________________
# batch_normalization_13 (BatchNo (None, 16, 16, 128)  512         add_6[0][0]
# __________________________________________________________________________________________________
# activation_13 (Activation)      (None, 16, 16, 128)  0           batch_normalization_13[0][0]
# __________________________________________________________________________________________________
# conv2d_15 (Conv2D)              (None, 16, 16, 128)  147584      activation_13[0][0]
# __________________________________________________________________________________________________
# batch_normalization_14 (BatchNo (None, 16, 16, 128)  512         conv2d_15[0][0]
# __________________________________________________________________________________________________
# activation_14 (Activation)      (None, 16, 16, 128)  0           batch_normalization_14[0][0]
# __________________________________________________________________________________________________
# conv2d_16 (Conv2D)              (None, 16, 16, 128)  147584      activation_14[0][0]
# __________________________________________________________________________________________________
# add_7 (Add)                     (None, 16, 16, 128)  0           add_6[0][0]
#                                                                  conv2d_16[0][0]
# __________________________________________________________________________________________________
# batch_normalization_15 (BatchNo (None, 16, 16, 128)  512         add_7[0][0]
# __________________________________________________________________________________________________
# activation_15 (Activation)      (None, 16, 16, 128)  0           batch_normalization_15[0][0]
# __________________________________________________________________________________________________
# conv2d_17 (Conv2D)              (None, 8, 8, 256)    295168      activation_15[0][0]
# __________________________________________________________________________________________________
# batch_normalization_16 (BatchNo (None, 8, 8, 256)    1024        conv2d_17[0][0]
# __________________________________________________________________________________________________
# activation_16 (Activation)      (None, 8, 8, 256)    0           batch_normalization_16[0][0]
# __________________________________________________________________________________________________
# conv2d_19 (Conv2D)              (None, 8, 8, 256)    33024       add_7[0][0]
# __________________________________________________________________________________________________
# conv2d_18 (Conv2D)              (None, 8, 8, 256)    590080      activation_16[0][0]
# __________________________________________________________________________________________________
# add_8 (Add)                     (None, 8, 8, 256)    0           conv2d_19[0][0]
#                                                                  conv2d_18[0][0]
# __________________________________________________________________________________________________
# batch_normalization_17 (BatchNo (None, 8, 8, 256)    1024        add_8[0][0]
# __________________________________________________________________________________________________
# activation_17 (Activation)      (None, 8, 8, 256)    0           batch_normalization_17[0][0]
# __________________________________________________________________________________________________
# conv2d_20 (Conv2D)              (None, 8, 8, 256)    590080      activation_17[0][0]
# __________________________________________________________________________________________________
# batch_normalization_18 (BatchNo (None, 8, 8, 256)    1024        conv2d_20[0][0]
# __________________________________________________________________________________________________
# activation_18 (Activation)      (None, 8, 8, 256)    0           batch_normalization_18[0][0]
# __________________________________________________________________________________________________
# conv2d_21 (Conv2D)              (None, 8, 8, 256)    590080      activation_18[0][0]
# __________________________________________________________________________________________________
# add_9 (Add)                     (None, 8, 8, 256)    0           add_8[0][0]
#                                                                  conv2d_21[0][0]
# __________________________________________________________________________________________________
# batch_normalization_19 (BatchNo (None, 8, 8, 256)    1024        add_9[0][0]
# __________________________________________________________________________________________________
# activation_19 (Activation)      (None, 8, 8, 256)    0           batch_normalization_19[0][0]
# __________________________________________________________________________________________________
# conv2d_22 (Conv2D)              (None, 8, 8, 256)    590080      activation_19[0][0]
# __________________________________________________________________________________________________
# batch_normalization_20 (BatchNo (None, 8, 8, 256)    1024        conv2d_22[0][0]
# __________________________________________________________________________________________________
# activation_20 (Activation)      (None, 8, 8, 256)    0           batch_normalization_20[0][0]
# __________________________________________________________________________________________________
# conv2d_23 (Conv2D)              (None, 8, 8, 256)    590080      activation_20[0][0]
# __________________________________________________________________________________________________
# add_10 (Add)                    (None, 8, 8, 256)    0           add_9[0][0]
#                                                                  conv2d_23[0][0]
# __________________________________________________________________________________________________
# batch_normalization_21 (BatchNo (None, 8, 8, 256)    1024        add_10[0][0]
# __________________________________________________________________________________________________
# activation_21 (Activation)      (None, 8, 8, 256)    0           batch_normalization_21[0][0]
# __________________________________________________________________________________________________
# conv2d_24 (Conv2D)              (None, 8, 8, 256)    590080      activation_21[0][0]
# __________________________________________________________________________________________________
# batch_normalization_22 (BatchNo (None, 8, 8, 256)    1024        conv2d_24[0][0]
# __________________________________________________________________________________________________
# activation_22 (Activation)      (None, 8, 8, 256)    0           batch_normalization_22[0][0]
# __________________________________________________________________________________________________
# conv2d_25 (Conv2D)              (None, 8, 8, 256)    590080      activation_22[0][0]
# __________________________________________________________________________________________________
# add_11 (Add)                    (None, 8, 8, 256)    0           add_10[0][0]
#                                                                  conv2d_25[0][0]
# __________________________________________________________________________________________________
# batch_normalization_23 (BatchNo (None, 8, 8, 256)    1024        add_11[0][0]
# __________________________________________________________________________________________________
# activation_23 (Activation)      (None, 8, 8, 256)    0           batch_normalization_23[0][0]
# __________________________________________________________________________________________________
# conv2d_26 (Conv2D)              (None, 8, 8, 256)    590080      activation_23[0][0]
# __________________________________________________________________________________________________
# batch_normalization_24 (BatchNo (None, 8, 8, 256)    1024        conv2d_26[0][0]
# __________________________________________________________________________________________________
# activation_24 (Activation)      (None, 8, 8, 256)    0           batch_normalization_24[0][0]
# __________________________________________________________________________________________________
# conv2d_27 (Conv2D)              (None, 8, 8, 256)    590080      activation_24[0][0]
# __________________________________________________________________________________________________
# add_12 (Add)                    (None, 8, 8, 256)    0           add_11[0][0]
#                                                                  conv2d_27[0][0]
# __________________________________________________________________________________________________
# batch_normalization_25 (BatchNo (None, 8, 8, 256)    1024        add_12[0][0]
# __________________________________________________________________________________________________
# activation_25 (Activation)      (None, 8, 8, 256)    0           batch_normalization_25[0][0]
# __________________________________________________________________________________________________
# conv2d_28 (Conv2D)              (None, 8, 8, 256)    590080      activation_25[0][0]
# __________________________________________________________________________________________________
# batch_normalization_26 (BatchNo (None, 8, 8, 256)    1024        conv2d_28[0][0]
# __________________________________________________________________________________________________
# activation_26 (Activation)      (None, 8, 8, 256)    0           batch_normalization_26[0][0]
# __________________________________________________________________________________________________
# conv2d_29 (Conv2D)              (None, 8, 8, 256)    590080      activation_26[0][0]
# __________________________________________________________________________________________________
# add_13 (Add)                    (None, 8, 8, 256)    0           add_12[0][0]
#                                                                  conv2d_29[0][0]
# __________________________________________________________________________________________________
# batch_normalization_27 (BatchNo (None, 8, 8, 256)    1024        add_13[0][0]
# __________________________________________________________________________________________________
# activation_27 (Activation)      (None, 8, 8, 256)    0           batch_normalization_27[0][0]
# __________________________________________________________________________________________________
# conv2d_30 (Conv2D)              (None, 4, 4, 512)    1180160     activation_27[0][0]
# __________________________________________________________________________________________________
# batch_normalization_28 (BatchNo (None, 4, 4, 512)    2048        conv2d_30[0][0]
# __________________________________________________________________________________________________
# activation_28 (Activation)      (None, 4, 4, 512)    0           batch_normalization_28[0][0]
# __________________________________________________________________________________________________
# conv2d_32 (Conv2D)              (None, 4, 4, 512)    131584      add_13[0][0]
# __________________________________________________________________________________________________
# conv2d_31 (Conv2D)              (None, 4, 4, 512)    2359808     activation_28[0][0]
# __________________________________________________________________________________________________
# add_14 (Add)                    (None, 4, 4, 512)    0           conv2d_32[0][0]
#                                                                  conv2d_31[0][0]
# __________________________________________________________________________________________________
# batch_normalization_29 (BatchNo (None, 4, 4, 512)    2048        add_14[0][0]
# __________________________________________________________________________________________________
# activation_29 (Activation)      (None, 4, 4, 512)    0           batch_normalization_29[0][0]
# __________________________________________________________________________________________________
# conv2d_33 (Conv2D)              (None, 4, 4, 512)    2359808     activation_29[0][0]
# __________________________________________________________________________________________________
# batch_normalization_30 (BatchNo (None, 4, 4, 512)    2048        conv2d_33[0][0]
# __________________________________________________________________________________________________
# activation_30 (Activation)      (None, 4, 4, 512)    0           batch_normalization_30[0][0]
# __________________________________________________________________________________________________
# conv2d_34 (Conv2D)              (None, 4, 4, 512)    2359808     activation_30[0][0]
# __________________________________________________________________________________________________
# add_15 (Add)                    (None, 4, 4, 512)    0           add_14[0][0]
#                                                                  conv2d_34[0][0]
# __________________________________________________________________________________________________
# batch_normalization_31 (BatchNo (None, 4, 4, 512)    2048        add_15[0][0]
# __________________________________________________________________________________________________
# activation_31 (Activation)      (None, 4, 4, 512)    0           batch_normalization_31[0][0]
# __________________________________________________________________________________________________
# conv2d_35 (Conv2D)              (None, 4, 4, 512)    2359808     activation_31[0][0]
# __________________________________________________________________________________________________
# batch_normalization_32 (BatchNo (None, 4, 4, 512)    2048        conv2d_35[0][0]
# __________________________________________________________________________________________________
# activation_32 (Activation)      (None, 4, 4, 512)    0           batch_normalization_32[0][0]
# __________________________________________________________________________________________________
# conv2d_36 (Conv2D)              (None, 4, 4, 512)    2359808     activation_32[0][0]
# __________________________________________________________________________________________________
# add_16 (Add)                    (None, 4, 4, 512)    0           add_15[0][0]
#                                                                  conv2d_36[0][0]
# __________________________________________________________________________________________________
# batch_normalization_33 (BatchNo (None, 4, 4, 512)    2048        add_16[0][0]
# __________________________________________________________________________________________________
# activation_33 (Activation)      (None, 4, 4, 512)    0           batch_normalization_33[0][0]
# __________________________________________________________________________________________________
# decoder_stage0_upsample (UpSamp (None, 8, 8, 512)    0           activation_33[0][0]
# __________________________________________________________________________________________________
# concatenate_1 (Concatenate)     (None, 8, 8, 768)    0           decoder_stage0_upsample[0][0]
#                                                                  activation_27[0][0]
# __________________________________________________________________________________________________
# decoder_stage0_conv1 (Conv2D)   (None, 8, 8, 256)    1769728     concatenate_1[0][0]
# __________________________________________________________________________________________________
# decoder_stage0_relu1 (Activatio (None, 8, 8, 256)    0           decoder_stage0_conv1[0][0]
# __________________________________________________________________________________________________
# decoder_stage0_conv2 (Conv2D)   (None, 8, 8, 256)    590080      decoder_stage0_relu1[0][0]
# __________________________________________________________________________________________________
# decoder_stage0_relu2 (Activatio (None, 8, 8, 256)    0           decoder_stage0_conv2[0][0]
# __________________________________________________________________________________________________
# decoder_stage1_upsample (UpSamp (None, 16, 16, 256)  0           decoder_stage0_relu2[0][0]
# __________________________________________________________________________________________________
# concatenate_2 (Concatenate)     (None, 16, 16, 384)  0           decoder_stage1_upsample[0][0]
#                                                                  activation_15[0][0]
# __________________________________________________________________________________________________
# decoder_stage1_conv1 (Conv2D)   (None, 16, 16, 128)  442496      concatenate_2[0][0]
#  __________________________________________________________________________________________________
# decoder_stage1_conv2 (Conv2D)   (None, 16, 16, 128)  147584      decoder_stage1_relu1[0][0]
# __________________________________________________________________________________________________
# # decoder_stage1_relu1 (Activatio (None, 16, 16, 128)  0           decoder_stage1_conv1[0][0]
# #
# __________________________________________________________________________________________________
# decoder_stage1_relu2 (Activatio (None, 16, 16, 128)  0           decoder_stage1_conv2[0][0]
# __________________________________________________________________________________________________
# decoder_stage2_upsample (UpSamp (None, 32, 32, 128)  0           decoder_stage1_relu2[0][0]
# __________________________________________________________________________________________________
# concatenate_3 (Concatenate)     (None, 32, 32, 192)  0           decoder_stage2_upsample[0][0]
#                                                                  activation_7[0][0]
# __________________________________________________________________________________________________
# decoder_stage2_conv1 (Conv2D)   (None, 32, 32, 64)   110656      concatenate_3[0][0]
# __________________________________________________________________________________________________
# decoder_stage2_relu1 (Activatio (None, 32, 32, 64)   0           decoder_stage2_conv1[0][0]
# __________________________________________________________________________________________________
# decoder_stage2_conv2 (Conv2D)   (None, 32, 32, 64)   36928       decoder_stage2_relu1[0][0]
# __________________________________________________________________________________________________
# decoder_stage2_relu2 (Activatio (None, 32, 32, 64)   0           decoder_stage2_conv2[0][0]
# __________________________________________________________________________________________________
# decoder_stage3_upsample (UpSamp (None, 64, 64, 64)   0           decoder_stage2_relu2[0][0]
# __________________________________________________________________________________________________
# decoder_stage3_conv1 (Conv2D)   (None, 64, 64, 32)   18464       decoder_stage3_upsample[0][0]
# __________________________________________________________________________________________________
# decoder_stage3_relu1 (Activatio (None, 64, 64, 32)   0           decoder_stage3_conv1[0][0]
# __________________________________________________________________________________________________
# decoder_stage3_conv2 (Conv2D)   (None, 64, 64, 32)   9248        decoder_stage3_relu1[0][0]
# __________________________________________________________________________________________________
# decoder_stage3_relu2 (Activatio (None, 64, 64, 32)   0           decoder_stage3_conv2[0][0]
# __________________________________________________________________________________________________
# decoder_stage4_upsample (UpSamp (None, 128, 128, 32) 0           decoder_stage3_relu2[0][0]
# __________________________________________________________________________________________________
# decoder_stage4_conv1 (Conv2D)   (None, 128, 128, 16) 4624        decoder_stage4_upsample[0][0]
# __________________________________________________________________________________________________
# decoder_stage4_relu1 (Activatio (None, 128, 128, 16) 0           decoder_stage4_conv1[0][0]
# __________________________________________________________________________________________________
# decoder_stage4_conv2 (Conv2D)   (None, 128, 128, 16) 2320        decoder_stage4_relu1[0][0]
# __________________________________________________________________________________________________
# decoder_stage4_relu2 (Activatio (None, 128, 128, 16) 0           decoder_stage4_conv2[0][0]
# __________________________________________________________________________________________________
# final_conv (Conv2D)             (None, 128, 128, 1)  145         decoder_stage4_relu2[0][0]
# __________________________________________________________________________________________________
# sigmoid (Activation)            (None, 128, 128, 1)  0           final_conv[0][0]
# ==================================================================================================
# Total params: 24,432,625
# Trainable params: 24,417,393
# Non-trainable params: 15,232
# __________________________________________________________________________________________________