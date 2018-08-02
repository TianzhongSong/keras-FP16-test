from keras.layers import Activation, Reshape, Permute, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Deconv2D, Conv2DTranspose
from keras.models import Model
from keras.initializers import orthogonal, constant, he_normal
from keras.regularizers import l2
from utils.keras_lrn import LRN2D
from math import ceil
import numpy as np


def get_deconv_filter(f_shape):
    """
    reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    """
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
      for y in range(heigh):
          value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
          bilinear[x, y] = value
    weights = np.zeros(f_shape, dtype=np.float32)
    for i in range(f_shape[2]):
      weights[:, :, i, i] = bilinear
    return weights


def SegNet(nClasses, input_height=360, input_width=480):
    img_input = Input(shape=(input_height, input_width, 3))
    kernel_size = 7
    weight_decay = 1e-4
    # encoder
    x = LRN2D(k=1)(img_input)
    x = Conv2D(64, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 180x240
    x = Conv2D(64, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 90x120
    x = Conv2D(64, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 45x60
    x = Conv2D(64, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 22x30

    # decoder
    deconv_filters = get_deconv_filter([2, 2, 64, 64])
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',
                        kernel_initializer=constant(deconv_filters))(x)
    x = Conv2D(64, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',
                        kernel_initializer=constant(deconv_filters))(x)
    x = ZeroPadding2D(padding=(1, 0))(x)
    x = Conv2D(64, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',
                        kernel_initializer=constant(deconv_filters))(x)
    x = Conv2D(64, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',
                        kernel_initializer=constant(deconv_filters))(x)
    x = Conv2D(64, (kernel_size, kernel_size), padding='same',
               kernel_initializer=orthogonal())(x)
    x = BatchNormalization()(x)

    x = Conv2D(nClasses, (1, 1), padding='valid',
               kernel_initializer=he_normal(), kernel_regularizer=l2(0.005))(x)
    x = Reshape((nClasses, input_height * input_width))(x)
    x = Permute((2, 1))(x)

    x = Activation('softmax')(x)
    model = Model(img_input, x, name='SegNet')
    return model
