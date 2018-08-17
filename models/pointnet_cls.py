from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization, Dense
from keras.layers import Reshape, Lambda, UpSampling2D
from keras.models import Model
import numpy as np
import tensorflow as tf


def mat_mul(A, B):
    return tf.matmul(A, B)


def PointNet(nb_classes):
    input_points = Input(shape=(2048, 3))
    # input transformation net
    x = Conv1D(64, 1, activation='relu')(input_points)
    x = BatchNormalization()(x)
    x = Conv1D(128, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(512, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2048)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)

    x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)

    input_T = Reshape((3, 3))(x)

    # forward net
    g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
    g = Conv1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # feature transform net
    f = Conv1D(64, 1, activation='relu')(g)
    f = BatchNormalization()(f)
    f = Conv1D(128, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Conv1D(256, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Conv1D(512, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Conv1D(1024, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = MaxPooling1D(pool_size=2048)(f)
    f = Dense(512, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = Reshape((64, 64))(f)

    # forward net
    g = Lambda(mat_mul, arguments={'B': feature_T})(g)
    g = Conv1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(128, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(256, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(512, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(1024, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # global feature
    global_feature = MaxPooling1D(pool_size=2048)(g)

    # point_net_cls
    c = Dense(512, activation='relu')(global_feature)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    c = Dense(256, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    c = Dense(nb_classes, activation='softmax')(c)
    prediction = Flatten()(c)

    model = Model(inputs=input_points, outputs=prediction)

    return model
