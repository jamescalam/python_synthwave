import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from datetime import datetime
import os


now = datetime.now().strftime('%m%d')
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class Generator:
    def __init__(self, latent_units, width, height):
        self.latent_units = latent_units
        self.width = width
        self.height = height

    def mnist_dcgan(self):
        """
        We use this model for benchmarking purposes, requires the mnist digits
        dataset.
        """
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(7 * 7 * 32, use_bias=False, input_shape=(100,)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())
        print(f"1: {self.model.output_shape}")

        self.model.add(tf.keras.layers.Reshape((7, 7, 32)))
        print(f"2: {self.model.output_shape}")

        self.model.add(tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())
        print(f"3: {self.model.output_shape}")

        self.model.add(tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())
        print(f"4: {self.model.output_shape}")

        self.model.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert self.model.output_shape == (None, 28, 28, 1)  # None = batchsize

    def optimiser(self, learning_rate):
        self.opt = tf.optimizers.Adam(learning_rate)

    def loss(self, fake_preds):
        # calculate the loss with binary cross-entropy
        fake_loss = cross_entropy(tf.ones_like(fake_preds), fake_preds)
        # return the fake predictions loss
        return fake_loss


class Discriminator:
    def __init__(self):
        pass

    def mnist_dcgan(self):
        """
        We use this model for benchmarking purposes, requires the mnist digits
        dataset.
        """
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[28, 28, 1]))
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(1))

    def optimiser(self, learning_rate):
        self.opt = tf.optimizers.Adam(learning_rate)

    def loss(self, real_preds, fake_preds):
        # take sigmoid of our output predictions
        #real_preds = tf.sigmoid(real_preds)
        #fake_preds = tf.sigmoid(fake_preds)
        # calculate the loss with binary cross-entropy
        real_loss = cross_entropy(tf.ones_like(real_preds), real_preds)
        fake_loss = cross_entropy(tf.zeros_like(fake_preds), fake_preds)
        # return the total loss from both real and fake predictions
        return real_loss + fake_loss


class Train:
    def __init__(self, G, D, width, height, batchsize, latent_units):
        self.G = G
        self.D = D
        self.width = width
        self.height = height
        self.batchsize = batchsize
        self.latent_units = latent_units
        self.history = pd.DataFrame({
            'gen_loss': [],
            'disc_loss': []
        })


    def step(self, sequences):
        # create noise
        #noise = np.random.randn(self.batchsize, self.latent_units)
        noise = tf.random.normal([self.batchsize, self.latent_units])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # generate fake sequences with the generator model
            generated_sequences = self.G.model(noise)

            # get real and fake output predictions from the discriminator model
            real_output = self.D.model(sequences, training=True)
            fake_output = self.D.model(generated_sequences, training=True)

            # get the loss for both the generator and discriminator models
            gen_loss = self.G.loss(fake_output)
            disc_loss = self.D.loss(real_output, fake_output)

            # get the gradients for both the generator and discriminator models
            grads_generator = gen_tape.gradient(
                gen_loss, self.G.model.trainable_variables
            )
            grads_discriminator = disc_tape.gradient(
                disc_loss, self.D.model.trainable_variables
            )

            self.G.opt.apply_gradients(
                zip(
                    grads_generator, self.G.model.trainable_variables
                )
            )

            self.D.opt.apply_gradients(
                zip(
                    grads_discriminator, self.D.model.trainable_variables
                )
            )

            self.history = self.history.append(pd.DataFrame({
                'gen_loss': [np.mean(gen_loss)],
                'disc_loss': [np.mean(disc_loss)]
            }), ignore_index=True)

            self.sequences = {
                'generated': generated_sequences,
                'true': sequences
            }

    def mod_step(self, sequences, i=0, start=1):
        # create noise
        noise = np.random.randn(self.batchsize, self.latent_units)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # generate fake sequences with the generator model
            generated_sequences = self.G.model(noise)
            # self.generated_sequences = tf.reshape(generated_sequences,
            #                                 shape=(self.batchsize,
            #                                        self.width,
            #                                        self.height,
            #                                        1))
            # get real and fake output predictions from the discriminator model
            real_output = self.D.model(sequences, training=True)
            fake_output = self.D.model(generated_sequences, training=True)

            # get the loss for both the generator and discriminator models
            gen_loss = self.G.loss(fake_output)
            disc_loss = self.D.loss(real_output, fake_output)

            # get the gradients for both the generator and discriminator models
            grads_generator = gen_tape.gradient(
                gen_loss, self.G.model.trainable_variables
            )
            grads_discriminator = disc_tape.gradient(
                disc_loss, self.D.model.trainable_variables
            )

            self.history = self.history.append(pd.DataFrame({
                'gen_loss': [np.mean(gen_loss)],
                'disc_loss': [np.mean(disc_loss) - np.mean(gen_loss)]
            }), ignore_index=True)

            if i >= start and np.mean(gen_loss) > (np.mean(disc_loss) - np.mean(gen_loss)):
                self.G.opt.apply_gradients(
                    zip(
                        grads_generator, self.G.model.trainable_variables
                    )
                )
            elif i >= start:
                self.D.opt.apply_gradients(
                    zip(
                        grads_discriminator, self.D.model.trainable_variables
                    )
                )
            else:
                self.G.opt.apply_gradients(
                    zip(
                        grads_generator, self.G.model.trainable_variables
                    )
                )
                self.D.opt.apply_gradients(
                    zip(
                        grads_discriminator, self.D.model.trainable_variables
                    )
                )

            self.history = self.history.append(pd.DataFrame({
                'gen_loss': [np.mean(gen_loss)],
                'disc_loss': [np.mean(disc_loss) - np.mean(gen_loss)]
            }), ignore_index=True)

            self.sequences = {
                'generated': generated_sequences,
                'true': sequences
            }

    def fit(self, dataset, epochs, vis=False):
        # mkdir if it does not exist
        if not os.path.exists(f'../visuals/{now}'):
            os.mkdir(f'../visuals/{now}')

        # check how many data batches we have
        num_batches = 0
        for batch in dataset:
            num_batches += 1

        for e in range(epochs):
            # for every epoch, choose 10 random steps to visualise
            step_idx = np.random.randint(0, num_batches, 10)
            print(f"Epoch {e}")  # so we now the epoch number

            # run a step
            for j, sequences in enumerate(dataset):
                self.step(sequences)

                if j in step_idx:
                    # now we visualise the results of this step, saving to file
                    # first, choose 16 random images from the batch
                    idx = np.random.randint(0, self.batchsize, 16)
                    # initialise matplotlib object
                    fig, ax = plt.subplots(4, 4, figsize=(3*4, 3*4))
                    i = 0  # initialise visuals indexer
                    for row in range(4):
                        for col in range(4):
                            # extract array
                            visual = self.sequences['generated'][idx[i]].numpy() * 127.5 + 127.5  # also adjust range to 0-255
                            # reshape array
                            visual = visual.reshape((visual.shape[0], visual.shape[1]))
                            # convert array to an image
                            image = Image.fromarray(visual)
                            # plot the image
                            ax[row, col].imshow(image, cmap='gray', vmin=0, vmax=255)
                            
                            i += 1

                    # save the figure to file
                    plt.savefig(f'../visuals/{now}/epoch_{e}_iter_{j}.jpg')

            print(f"gen_loss: {self.history['gen_loss'].iloc[len(self.history)-1]}\ndisc_loss: {self.history['disc_loss'].iloc[len(self.history)-1]}")


            if vis:
                plt.clf()
                fig = plt.figure(figsize=(18, 12))

                sns.lineplot(x=range(len(self.history)), y=self.history['gen_loss'],
                                        label='G', color='#5AFFE7')
                sns.lineplot(x=range(len(self.history)), y=self.history['disc_loss'],
                                        label='D', color='#726EFF')
                plt.show()

    def show(self, array):
        fig = plt.figure(figsize=(18, 12))
        image = Image.fromarray(array)
        plt.imshow(image)
        plt.show()