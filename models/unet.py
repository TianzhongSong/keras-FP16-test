from keras.models import Model
from keras.layers import Input, concatenate, Dropout, Reshape, Permute, Activation, ZeroPadding2D, Cropping2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
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
    f = ceil(width / 2.0)
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


def Unet(nClasses, input_height=360, input_width=480, nChannels=3):
    inputs = Input(shape=(input_height, input_width, nChannels))
    # encode
    # 360x480
    x = LRN2D(k=1)(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(x)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 180x240
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 90x120
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 45x60
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 22x30
    # decode
    deconv_filters = get_deconv_filter([2, 2, 64, 64])
    up5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',
                          kernel_initializer=constant(deconv_filters))(pool4)
    up5 = ZeroPadding2D(padding=(1, 0))(up5)
    up5 = Cropping2D(cropping=((1, 0), (0, 0)))(up5)
    up5 = concatenate([up5, conv4], axis=-1)
    # 45x60
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up5)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv5)

    up6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',
                          kernel_initializer=constant(deconv_filters))(conv5)
    up6 = concatenate([up6, conv3], axis=-1)
    # 90x120
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv6)

    up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',
                          kernel_initializer=constant(deconv_filters))(conv6)
    up7 = concatenate([up7, conv2], axis=-1)
    # 180x240
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',
                          kernel_initializer=constant(deconv_filters))(conv7)
    up8 = concatenate([up8, conv1], axis=-1)
    # 360x480
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv8)

    conv9 = Conv2D(nClasses, (1, 1), activation='relu', padding='same',
                   kernel_initializer=he_normal(), kernel_regularizer=l2(0.005))(conv8)
    conv9 = Reshape((nClasses, input_height * input_width))(conv9)
    conv9 = Permute((2, 1))(conv9)

    conv9 = Activation('softmax')(conv9)

    model = Model(input=inputs, output=conv9)

    return model
