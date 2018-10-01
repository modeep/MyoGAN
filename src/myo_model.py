import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, Reshape, BatchNormalization, Dense, Activation, UpSampling2D, Input
from keras.layers import Flatten
from keras.layers.advanced_activations import LeakyReLU
from .load_data import DataLoader_Continous

'''
Model structure

Input : (100) Vector
Output : (64, 64, 1) image

Generator
100 -> 256
256 -> 16 x 16
16 x 16 -> 32 x 32
32 x 32 -> 64 x 64
64 x 64 -> 128 x 128

Discriminator
128 x 128 -> 64 x 64
64 x 64 -> 32 x 32
32 x 32 -> 16 x 16
16 x 16 -> 8 x 8
8 x 8 -> 64
64 -> 32 (16, 10)

* Referenced from jigeria's code
'''

# TODO: 1. Conditional GAN
# TODO: 2. LSTM Input (EMG vs Noise)
# TODO: 3. Threshold (Activation)


class MyoGAN():
    def __init__(self):
        self.noise_size = 100
        self.noise_input = Input(shape=(self.noise_size,))
        self.image_size = 128
        self.image_channel = 1
        self.loader = DataLoader_Continous(data_path='./dataset_2018_05_16/',
                                           is_real_image=False,
                                           data_type=2,
                                           emg_length=600,
                                           is_flatten=False)

    def generative(self):
        _ = (Dense(256, input_shape=(100,), activation='relu'))
        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Reshape((16, 16, 1), input_shape=(256,))(_)

        _ = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', input_shape=(16, 16, 1))(
            _)
        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Activation(activation='relu')(_)

        _ = UpSampling2D()(_)
        _ = Conv2D(filters=256, kernel_size=3, padding='same', input_shape=(16, 16, 128))(_)
        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Activation(activation='relu')(_)

        _ = UpSampling2D()(_)
        _ = Conv2D(filters=512, kernel_size=3, padding='same', input_shape=(32, 32, 256))(_)
        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Activation(activation='relu')(_)

        _ = UpSampling2D()(_)
        _ = Conv2D(filters=256, kernel_size=3, padding='same', input_shape=(64, 64, 512))(_)
        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Activation(activation='relu')(_)

        # _ = UpSampling2D()(_)
        _ = Conv2D(filters=1, kernel_size=3, padding='same', input_shape=(128, 128, 256))(_)
        _ = Activation(activation='tanh')(_)

        return Model(inputs=self.noise_input, outputs=_)

    def discriminative(self):
        _ = inputs = Input(shape=(self.image_size, self.image_size, self.image_channel))

        _ = Conv2D(filters=256, kernel_size=(2, 2), strides=2, padding='same', input_shape=(128, 128, 1))(_)
        _ = LeakyReLU(alpha=0.2)(_)

        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Conv2D(filters=512, kernel_size=(1, 1), strides=2, padding='same', input_shape=(64, 64, 256))(_)
        _ = LeakyReLU(alpha=0.2)(_)

        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Conv2D(filters=256, kernel_size=(2, 2), strides=2, padding='same', input_shape=(32, 32, 512))(_)
        _ = LeakyReLU(alpha=0.2)(_)

        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same', input_shape=(16, 16, 256))(_)
        _ = LeakyReLU(alpha=0.2)(_)

        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same', input_shape=(8, 8, 128))(_)
        _ = LeakyReLU(alpha=0.2)(_)

        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Conv2D(filters=self.image_channel, kernel_size=(2, 2), strides=1, padding='same', input_shape=(4, 4, 128))(
            _)
        # _ = LeakyReLU(alpha=0.2)(_)

        outputs = Flatten()(_)
        outputs = Dense(1, activation='sigmoid')(outputs)

        return Model(inputs=inputs, outputs=outputs)

