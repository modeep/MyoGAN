import matplotlib.pyplot as plt
import numpy as np
from keras.layers import BatchNormalization, Activation
from keras.layers import Conv2D, UpSampling2D
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from load_data import DataLoader_Continous


class MyoLSGAN():
    def __init__(self):
        self.img_size = 128
        self.channels = 1
        self.img_shape = (self.img_size, self.img_size, self.channels)
        self.noise_size = 100
        self.d_step = 50
        self.g_step = 10
        self.g_loss_history = []
        self.d_loss_real_history = []
        self.d_loss_fake_history = []
        self.d_loss_history = []
        self.loader = DataLoader_Continous(data_path='./dataset_2018_05_16/',
                                           is_real_image=False,
                                           data_type=2,
                                           emg_length=600,
                                           is_flatten=False)

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.noise_size,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='mse', optimizer=optimizer)

    def build_generator(self):
        # model = Sequential()
        #
        # model.add(Dense(256, input_shape=(100,), activation='elu'))
        #
        # model.add(Reshape((16, 16, 1), input_shape=(256,)))
        #
        # model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', input_shape=(16, 16, 1)))
        # model.add(Activation(activation='elu'))
        #
        # model.add(UpSampling2D())
        # model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', input_shape=(16, 16, 128)))
        # model.add(Activation(activation='elu'))
        #
        # model.add(UpSampling2D())
        # model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', input_shape=(32, 32, 256)))
        # model.add(Activation(activation='elu'))
        #
        # model.add(UpSampling2D())
        # model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', input_shape=(64, 64, 512)))
        # model.add(Activation(activation='elu'))
        #
        # # _ = UpSampling2D()(_)
        # model.add(Conv2D(filters=1, kernel_size=(3, 3), strides=1, padding='same', input_shape=(128, 128, 256)))
        # # _ = Activation(activation='tanh')(_)
        # model.add(Activation(activation='tanh'))

        model = Sequential()

        model.add(Dense(256, input_shape=self.noise_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.noise_size,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        # model = Sequential()
        #
        # model.add(Conv2D(filters=256, kernel_size=(2, 2), strides=2, padding='same', input_shape=(128, 128, 1)))
        # model.add(LeakyReLU(alpha=0.2))
        #
        # model.add(BatchNormalization(axis=1))
        # model.add(Conv2D(filters=512, kernel_size=(1, 1), strides=2, padding='same', input_shape=(64, 64, 256)))
        # model.add(LeakyReLU(alpha=0.2))
        #
        # model.add(BatchNormalization(axis=1))
        # model.add(Conv2D(filters=256, kernel_size=(2, 2), strides=2, padding='same', input_shape=(32, 32, 512)))
        # model.add(LeakyReLU(alpha=0.2))
        #
        # model.add(BatchNormalization(axis=1))
        # model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same', input_shape=(16, 16, 256)))
        # model.add(LeakyReLU(alpha=0.2))
        #
        # model.add(BatchNormalization(axis=1))
        # model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same', input_shape=(8, 8, 128)))
        # model.add(LeakyReLU(alpha=0.2))
        #
        # model.add(BatchNormalization(axis=1))
        # model.add(Conv2D(filters=self.channels, kernel_size=(2, 2), strides=1, padding='same', input_shape=(4, 4, 128)))

        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=32, sample_interval=5):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            images = self.loader.get_images(batch_size)

            for _ in range(self.d_step):
                noise = np.random.normal(0, 1, (batch_size, self.noise_size))
                g_z = self.generator.predict(noise)

                d_loss_real = self.discriminator.train_on_batch(images, valid)
                d_loss_fake = self.discriminator.train_on_batch(g_z, fake)
                d_loss = .5 * np.add(d_loss_fake, d_loss_real)

                self.d_loss_real_history.append(d_loss_real)
                self.d_loss_fake_history.append(d_loss_fake)
                self.d_loss_history.append(d_loss)

            for _ in range(self.g_step):
                noise = np.random.normal(0, 1, (batch_size, self.noise_size))
                g_loss = self.combined.train_on_batch(noise, valid)
                self.g_loss_history.append(g_loss)

            print('[Epoch {0}] D_loss_real: {1}  D_loss_fake: {2}  acc: {3}  G_loss: {4}'
                  .format(epoch, self.d_loss_real_history[-1], self.d_loss_fake_history[-1],
                          self.d_loss_history[-1][1] * 100, self.g_loss_history[-1]))

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
    gan = MyoLSGAN()
    gan.train(epochs=5000)
