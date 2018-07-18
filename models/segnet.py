from keras.layers import Activation, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model


def SegNet(nClasses, input_height=360, input_width=480):
    img_input = Input(shape=(input_height, input_width, 3))
    # encoder
    x = Conv2D(64, (3, 3), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # decoder
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(nClasses, (1, 1), padding='valid')(x)
    x = Reshape((nClasses, 352 * 480))(x)
    x = Permute((2, 1))(x)

    x = Activation('softmax')(x)
    model = Model(img_input, x, name='SegNet')
    return model
