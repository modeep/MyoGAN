import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization
from keras.layers import Conv2D, UpSampling2D, Activation
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf


class TestGAN:
    def __init__(self):
        self.img_size = 128
        self.channels = 1
        self.img_shape = (self.img_size, self.img_size, self.channels)
        self.noise_size = 100

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.noise_size,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined_model = Model(z, valid)
        self.combined_model.compile(loss='mse', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(256, input_shape=(100,)))
        model.add(BatchNormalization(axis=1))
        model.add(LeakyReLU())
        model.add(Reshape((16, 16, 1), input_shape=(256,)))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', input_shape=(16, 16, 1)))
        model.add(BatchNormalization(axis=1))
        model.add(LeakyReLU())

        model.add(UpSampling2D())
        model.add(Conv2D(filters=256, kernel_size=3, padding='same', input_shape=(16, 16, 128)))
        model.add(BatchNormalization(axis=1))
        model.add(LeakyReLU())

        model.add(UpSampling2D())
        model.add(Conv2D(filters=512, kernel_size=3, padding='same', input_shape=(32, 32, 256)))
        model.add(BatchNormalization(axis=1))
        model.add(LeakyReLU())

        model.add(UpSampling2D())
        model.add(Conv2D(filters=256, kernel_size=3, padding='same', input_shape=(64, 64, 512)))
        model.add(BatchNormalization(axis=1))
        model.add(LeakyReLU())

        model.add(Conv2D(filters=1, kernel_size=3, padding='same', input_shape=(128, 128, 256)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation(activation='tanh'))

        model.summary()

        noise = Input(shape=(self.noise_size,))
        img = model(noise)

        return Model(inputs=noise, outputs=img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(filters=256, kernel_size=(2, 2), strides=2, padding='same', input_shape=(128, 128, 1)))
        model.add(LeakyReLU())

        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(filters=512, kernel_size=(2, 2), strides=2, padding='same', input_shape=(64, 64, 256)))
        model.add(LeakyReLU())

        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(filters=256, kernel_size=(2, 2), strides=2, padding='same', input_shape=(32, 32, 512)))
        model.add(LeakyReLU())

        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same', input_shape=(16, 16, 256)))
        model.add(LeakyReLU())

        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=2, padding='same', input_shape=(8, 8, 128)))
        model.add(LeakyReLU())

        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(filters=self.channels, kernel_size=(2, 2), strides=1, padding='same', input_shape=(4, 4, 64)))
        model.add(LeakyReLU())

        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs=30000, batch_size=32, sample_interval=5):
        (x_train, _), (_, _) = mnist.load_data()

        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_train = np.expand_dims(x_train, axis=3)
        x_train = tf.image.resize_images(x_train, [128, 128])

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # Train Discriminator

            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.noise_size))

            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator

            g_loss = self.combined_model.train_on_batch(noise, valid)

            print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.noise_size))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig('./model_output/image/fake_image{0}.png'.format(epoch))
        plt.close()


if __name__ == '__main__':
    gan = TestGAN()
    gan.train(epochs=5000)