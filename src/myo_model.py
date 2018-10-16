import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Reshape, BatchNormalization, Dense, Activation, UpSampling2D, Input
from keras.layers import Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.models import model_from_json
from keras.optimizers import Adam
from load_data import DataLoader_Continous

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


class MyoGAN:
    def __init__(self):
        self.g_loss_history = []
        self.d_loss_real_history = []
        self.d_loss_fake_history = []

        self.d_step = 1
        self.g_step = 2
        self.epoch = 30000
        self.batch_size = 64
        self.noise_size = 100
        self.noise_input = Input(shape=(self.noise_size,))
        self.image_size = 128
        self.image_channel = 1

        self.loader = DataLoader_Continous(data_path='./dataset_2018_05_16/',
                                           is_real_image=False,
                                           data_type=2,
                                           emg_length=600,
                                           is_flatten=False)

        self.net_d = self.discriminative()
        self.net_g = self.generative()
        fake_image = self.net_g(self.noise_input)

        combined_output = self.net_d(fake_image)
        self.combined_model = Model(inputs=[self.noise_input], outputs=[combined_output], name='combined')

        adam = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
        self.net_g.compile(loss='binary_crossentropy', optimizer=adam)
        self.net_d.compile(loss='binary_crossentropy', optimizer=adam)
        self.net_d.trainable = False
        self.combined_model.compile(loss='binary_crossentropy', optimizer=adam)

        self.combined_model.summary()

    # def generative(self):
    #     _ = Dense(256, input_shape=(100,), activation='relu')(self.noise_input)
    #     _ = BatchNormalization(axis=1)(_, training=1)
    #     _ = Reshape((16, 16, 1), input_shape=(256,))(_)
    #
    #     _ = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', input_shape=(16, 16, 1))(
    #         _)
    #     _ = BatchNormalization(axis=1)(_, training=1)
    #     _ = Activation(activation='relu')(_)
    #
    #     _ = UpSampling2D()(_)
    #     _ = Conv2D(filters=256, kernel_size=3, padding='same', input_shape=(16, 16, 128))(_)
    #     _ = BatchNormalization(axis=1)(_, training=1)
    #     _ = Activation(activation='relu')(_)
    #
    #     _ = UpSampling2D()(_)
    #     _ = Conv2D(filters=512, kernel_size=3, padding='same', input_shape=(32, 32, 256))(_)
    #     _ = BatchNormalization(axis=1)(_, training=1)
    #     _ = Activation(activation='relu')(_)
    #
    #     _ = UpSampling2D()(_)
    #     _ = Conv2D(filters=256, kernel_size=3, padding='same', input_shape=(64, 64, 512))(_)
    #     _ = BatchNormalization(axis=1)(_, training=1)
    #     _ = Activation(activation='relu')(_)
    #
    #     # _ = UpSampling2D()(_)
    #     _ = Conv2D(filters=1, kernel_size=3, padding='same', input_shape=(128, 128, 256))(_)
    #     _ = Activation(activation='tanh')(_)
    #
    #     return Model(inputs=self.noise_input, outputs=_)

    def generative(self):
        _ = Dense(256, input_shape=(100,), activation='elu')(self.noise_input)
        _ = Reshape((16, 16, 1), input_shape=(256,))(_)

        _ = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', input_shape=(16, 16, 1))(_)
        _ = Activation(activation='elu')(_)

        _ = UpSampling2D()(_)
        _ = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', input_shape=(16, 16, 128))(_)
        _ = Activation(activation='elu')(_)

        _ = UpSampling2D()(_)
        _ = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', input_shape=(32, 32, 256))(_)
        _ = Activation(activation='elu')(_)

        _ = UpSampling2D()(_)
        _ = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', input_shape=(64, 64, 512))(_)
        _ = Activation(activation='elu')(_)

        # _ = UpSampling2D()(_)
        _ = Conv2D(filters=1, kernel_size=(3, 3), strides=1, padding='same', input_shape=(128, 128, 256))(_)
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

    def load_model(self):
        json_file = open('./model_output/load_model/g_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        load_g_model = model_from_json(loaded_model_json)
        # load weights into new model
        load_g_model.load_weights("./model_output/load_model/g_model.h5")

        json_file = open('./model_output/load_model/d_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        load_d_model = model_from_json(loaded_model_json)
        # load weights into new model
        load_d_model.load_weights("./model_output/load_model/d_model.h5")

        json_file = open('./model_output/load_model/combined_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        load_combined_model = model_from_json(loaded_model_json)
        # load weights into new model
        load_combined_model.load_weights("./model_output/load_model/combined_model.h5")

        print("Model loaded from disk!")

        return [load_g_model, load_d_model, load_combined_model]

    def sample_generation(self, num, net_g):
        for _ in range(num):
            noise = np.random.normal(size=[num, self.noise_size])
            gan_image = net_g.predict(noise)
            cv2.imwrite('./model_output/image/' + 'sample image' + str(_) + '.png', gan_image[_] * 127.5)

        print("generated image")

    def save_model(self):
        self.net_g.save_weights("./model_output/save_model/g_model.h5")
        self.net_d.save_weights("./model_output/save_model/d_model.h5")
        self.combined_model.save_weights("./model_output/save_model/combined_model.h5")

        g_model_json = self.net_g.to_json()
        with open("./model_output/save_model/g_model.json", "w") as json_file:
            json_file.write(g_model_json)

        d_model_json = self.net_d.to_json()
        with open("./model_output/save_model/d_model.json", "w") as json_file:
            json_file.write(d_model_json)

        combined_model_json = self.combined_model.to_json()
        with open("./model_output/save_model/combined_model.json", "w") as json_file:
            json_file.write(combined_model_json)

        print("Model saved to disk!")

    def train(self):
        i = 0

        while i <= self.epoch:
            # x_train = loader.get_emg_datas(batch_size)
            images = self.loader.get_images(self.batch_size)

            for _ in range(self.d_step):
                noise = np.random.normal(size=[self.batch_size, self.noise_size])

                g_z = self.net_g.predict(noise)

                d_loss_real = self.net_d.train_on_batch(images,
                                                        np.random.uniform(low=0.7, high=1.2, size=self.batch_size))
                d_loss_fake = self.net_d.train_on_batch(g_z, np.random.uniform(low=0.0, high=0.3, size=self.batch_size))

                self.d_loss_real_history.append(d_loss_real)
                self.d_loss_fake_history.append(d_loss_fake)
                self.d_loss = np.sum([d_loss_fake, d_loss_real])

            for _ in range(self.g_step):
                noise = np.random.normal(size=[self.batch_size, self.noise_size])
                combined_loss = self.combined_model.train_on_batch(noise, np.random.uniform(low=0.7, high=1.2,
                                                                                            size=self.batch_size))

                self.g_loss_history.append(combined_loss)

            print("%d [D loss real: %f] [D loss fake: %f] [G loss: %f]" % (
                i, self.d_loss_real_history[-1], self.d_loss_fake_history[-1], self.g_loss_history[-1]))

            # print("%d [D loss real: %f] [D loss fake: %f] [D loss: %f] [G loss: %f]" % (
            #     i, self.d_loss_real_history[-1], self.d_loss_fake_history[-1], self.d_loss, self.g_loss_history[-1]))

            if i % 500 == 0:
                gan_image = self.net_g.predict(np.random.normal(size=[self.batch_size, self.noise_size]))
                print("GAN Image 2: ", gan_image[0].shape)
                cv2.imwrite('./model_output/image/' + 'fake_image' + str(i) + '.png', gan_image[0] * 127.5)
                # cv2.imwrite('./output_image3/' + 'real_image'+ str(i) + '.png', images[0] * 127.5)

            i += 1

        self.sample_generation(32, self.net_g)

    def show_history(self):
        plt.figure(1, figsize=(16, 8))
        plt.plot(self.d_loss_real_history)
        plt.ylabel('d_loss_real')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')

        plt.savefig('./model3_output/image/d_loss_real_history.png')

        plt.figure(2, figsize=(16, 8))
        plt.plot(self.d_loss_fake_history)
        plt.ylabel('d_loss_fake')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')

        plt.savefig('./model3_output/image/d_loss_fake_history.png')

        # plt.show()

        plt.figure(3, figsize=(16, 8))
        plt.plot(self.g_loss_history)
        plt.ylabel('g_loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')

        plt.savefig('./model3_output/image/g_loss_history.png')

        # plt.show()


if __name__ == '__main__':
    myo_gan = MyoGAN()
    myo_gan.train()

    print("Finish!")
