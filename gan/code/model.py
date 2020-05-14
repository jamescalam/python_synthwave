import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from PIL import Image
from datetime import datetime
import os
import sys
sys.path.insert(0, '../../../notify/code')
import notify


now = datetime.now().strftime('%m%d')
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class Generator:
    def __init__(self, latent_units, width, height, depth=1):
        self.latent_units = latent_units
        self.width = width
        self.height = height
        self.depth = depth

    def mnist_dcgan(self):
        """
        We use this model for benchmarking purposes, requires the mnist digits
        dataset.
        """
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(7 * 7 * 32, use_bias=False, input_shape=(100,)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(tf.keras.layers.Reshape((7, 7, 32)))

        self.model.add(tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert self.model.output_shape == (None, 28, 28, 1)  # None = batchsize
        
    def dcgan(self):
        """
        This is our star Synthwave maker.
        
        Our final output size is 640 * 360, with 3 filters, so we use the
        following dimensions (x and y are always multipled/divided by 2).
        
        latent_variables
            V
        (80, 45), 32
            V
        (160, 90), 16
            V
        (320, 180), 8
            V
        (640, 360), 3
        """
        if self.width == 640:
            depth = 32
        elif self.width == 320:
            depth= 16
        else:
            raise ValueError("Incompatable width value provided:\n"
                             f"\twidth = {self.width}\n"
                             f"\theight = {self.height}")
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(80 * 45 * depth,
                                             use_bias=False,
                                             input_shape=(self.latent_units,)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(tf.keras.layers.Reshape((45, 80, depth)))

        self.model.add(
                tf.keras.layers.Conv2DTranspose(
                        int(depth/2),
                        (3, 3),
                        strides=(2, 2),
                        padding='same',
                        use_bias=False
                        )
                )
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        if self.model.output_shape[1] != self.height/2:
            self.model.add(
                    tf.keras.layers.Conv2DTranspose(
                            int(depth/4),
                            (3, 3),
                            strides=(2, 2),
                            padding='same',
                            use_bias=False
                            )
                    )
            self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(
                tf.keras.layers.Conv2DTranspose(
                        self.depth,
                        (3, 3),
                        strides=(2, 2),
                        padding='same',
                        use_bias=False,
                        activation='tanh'
                        )
                )
        print(self.model.output_shape)
        # check that our output shape matches that given by self.width and
        # self.height
        assert self.model.output_shape == (None, self.height, self.width, self.depth)

    def dcgan_v2(self):
        """
        This is our star Synthwave maker.
        
        Our final output size is 640 * 360, with 3 filters, so we use the
        following dimensions (x and y are always multipled/divided by 2).
        
        latent_variables
            V
        (80, 45), 32
            V
        (160, 90), 16
            V
        (320, 180), 8
            V
        (640, 360), 3
        """
        if self.width == 640:
            depth = 64
        elif self.width == 320:
            depth = 32
        else:
            raise ValueError("Incompatable width value provided:\n"
                             f"\twidth = {self.width}\n"
                             f"\theight = {self.height}")
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(80 * 45 * depth,
                                             use_bias=False,
                                             input_shape=(self.latent_units,)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(tf.keras.layers.Reshape((45, 80, depth)))

        # layer 1
        self.model.add(
                tf.keras.layers.Conv2DTranspose(
                        int(depth/2),
                        (3, 3),
                        strides=(1, 1),
                        padding='same',
                        use_bias=False
                        )
                )
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())
        
        self.model.add(
                tf.keras.layers.Conv2DTranspose(
                        int(depth/2),
                        (3, 3),
                        strides=(2, 2),
                        padding='same',
                        use_bias=False
                        )
                )
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        # layer 2 (if applicable)
        if self.model.output_shape[1] != self.height/2:
            self.model.add(
                    tf.keras.layers.Conv2DTranspose(
                            int(depth/4),
                            (3, 3),
                            strides=(1, 1),
                            padding='same',
                            use_bias=False
                            )
                    )
            self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.LeakyReLU())
            
            self.model.add(
                    tf.keras.layers.Conv2DTranspose(
                            int(depth/4),
                            (3, 3),
                            strides=(2, 2),
                            padding='same',
                            use_bias=False
                            )
                    )
            self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.LeakyReLU())

        # final layer
        self.model.add(
                tf.keras.layers.Conv2DTranspose(
                        self.depth,
                        (3, 3),
                        strides=(1, 1),
                        padding='same',
                        use_bias=False
                        )
                )
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())
        
        self.model.add(
                tf.keras.layers.Conv2DTranspose(
                        self.depth,
                        (3, 3),
                        strides=(2, 2),
                        padding='same',
                        use_bias=False,
                        activation='tanh'
                        )
                )
        print(self.model.output_shape)
        # check that our output shape matches that given by self.width and
        # self.height
        assert self.model.output_shape == (None, self.height, self.width, self.depth)
        
        
    def cyclegan(self):
        """
        This is our star Synthwave maker.
        
        Consumes an image (either (320, 180, 1) or (640, 360, 1)).
        
        Our final output size is 640*360 (or 320*180), with 3 filters, so we
        use the following dimensions (x and y are always multipled/divided by
        2).
        
        (640, 360) **
        (320, 180) conv
        (160, 90) conv
        (80, 45) conv-1
        (160, 90) conv-1
        (320, 180) conv-1
        (640, 360) **
        """
        depth = 16  # max number of filters at the deepest point
        # if a half resolution image is used this will not be applied, rather,
        # the max number of filters will be depth/2

        self.model = tf.keras.Sequential()

        # __________________ DOWMSAMPLING LAYERS __________________ #
        
        self.model.add(
                tf.keras.layers.Conv2D(
                        int(depth/4),
                        (3, 3),
                        strides=(2, 2),
                        padding='same',
                        use_bias=False,
                        input_shape=(self.latent_units[0], self.latent_units[1], 1)
                        )
                )
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())
        
        self.model.add(
                tf.keras.layers.Conv2D(
                        int(depth/2),
                        (3, 3),
                        strides=(2, 2),
                        padding='same',
                        use_bias=False
                        )
                )
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())
        
        if self.width == 640:
            self.model.add(
                tf.keras.layers.Conv2D(
                        depth,
                        (3, 3),
                        strides=(2, 2),
                        padding='same',
                        use_bias=False
                        )
                )
            self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.LeakyReLU())
            
        
        # ___________________ UPSAMPLING LAYERS ___________________ #

        if self.width == 640:
            # layer 1 (if applicable)            
            self.model.add(
                    tf.keras.layers.Conv2DTranspose(
                            int(depth/2),
                            (3, 3),
                            strides=(2, 2),
                            padding='same',
                            use_bias=False
                            )
                    )
            self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.LeakyReLU())

        # layer 2        
        self.model.add(
                tf.keras.layers.Conv2DTranspose(
                        int(depth/4),
                        (3, 3),
                        strides=(2, 2),
                        padding='same',
                        use_bias=False
                        )
                )
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        # final layer        
        self.model.add(
                tf.keras.layers.Conv2DTranspose(
                        self.depth,
                        (3, 3),
                        strides=(2, 2),
                        padding='same',
                        use_bias=False,
                        activation='tanh'
                        )
                )
        print(self.model.output_shape)
        # check that our output shape matches that given by self.width and
        # self.height
        assert self.model.output_shape == (None, self.height, self.width, self.depth)


    def optimiser(self, learning_rate):
        self.opt = tf.optimizers.Adam(learning_rate)

    def loss(self, fake_preds):
        # calculate the loss with binary cross-entropy
        fake_loss = cross_entropy(tf.ones_like(fake_preds), fake_preds)
        # return the fake predictions loss
        return fake_loss


class Discriminator:
    def __init__(self, width, height, depth=1):
        self.width = width
        self.height = height
        self.depth = depth

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
        
    def dcgan(self):
        """
        This is our star Synthwave detecting detective.
        """
        self.model = tf.keras.Sequential()
        self.model.add(
                tf.keras.layers.Conv2D(
                        8,
                        (5, 5),
                        strides=(2, 2),
                        padding='same',
                        input_shape=[self.height, self.width, self.depth]
                        )
                )
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(
                tf.keras.layers.Conv2D(
                        16,
                        (5, 5),
                        strides=(2, 2),
                        padding='same'
                        )
                )
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(0.5))
        
        if self.width == 640:
            self.model.add(
                    tf.keras.layers.Conv2D(
                            32,
                            (5, 5),
                            strides=(2, 2),
                            padding='same'
                            )
                    )
            self.model.add(tf.keras.layers.LeakyReLU())
            self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(1))
        
    def dcgan_v2(self):
        """
        This is our star Synthwave detecting detective.
        """
        self.model = tf.keras.Sequential()
        # layer 1 (2 CNNs)
        self.model.add(
                tf.keras.layers.Conv2D(
                        8,
                        (5, 5),
                        strides=(1, 1),
                        padding='same',
                        input_shape=[self.height, self.width, self.depth]
                        )
                )
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(
                tf.keras.layers.Conv2D(
                        8,
                        (3, 3),
                        strides=(2, 2),
                        padding='same'
                        )
                )
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(0.2))

        # layer 2
        self.model.add(
                tf.keras.layers.Conv2D(
                        16,
                        (3, 3),
                        strides=(1, 1),
                        padding='same'
                        )
                )
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(
                tf.keras.layers.Conv2D(
                        16,
                        (3, 3),
                        strides=(2, 2),
                        padding='same'
                        )
                )
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(0.2))
        
        # layer 3 (if applicable)
        if self.width == 640:
            self.model.add(
                    tf.keras.layers.Conv2D(
                            32,
                            (3, 3),
                            strides=(1, 1),
                            padding='same'
                            )
                    )
            self.model.add(tf.keras.layers.LeakyReLU())
            self.model.add(tf.keras.layers.Dropout(0.2))
            self.model.add(
                    tf.keras.layers.Conv2D(
                            32,
                            (3, 3),
                            strides=(2, 2),
                            padding='same'
                            )
                    )
            self.model.add(tf.keras.layers.LeakyReLU())
            self.model.add(tf.keras.layers.Dropout(0.2))

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


def save_gen_image(img_idx, plot_grid, filename, sequences, colour, figsize,
                   single=False):
    # initialise matplotlib object
    if figsize is None:
        fig, ax = plt.subplots(plot_grid, plot_grid,
                               figsize=(plot_grid*8, plot_grid*8))
    else:
        fig, ax = plt.subplots(plot_grid, plot_grid,
                               figsize=figsize)
    fig.set_tight_layout(True)

    if single:
        if single:
            visual = sequences.numpy() * 127.5 + 127.5  # adjust to 0-255
            visual = visual.reshape(visual.shape[1:])  # remove batch dim
            visual = visual.astype(np.uint8)  # convert to ints only
            visual = Image.fromarray(visual)
            visual.save(filename)  # save the image to file
            return
        
    i = 0  # initialise visuals indexer
    for row in range(plot_grid):
        for col in range(plot_grid):
            # extract array
            visual = sequences[img_idx[i]].numpy() * 127.5 + 127.5  # also adjust range to 0-255
            
            if not colour:
                # reshape array
                visual = visual.reshape((visual.shape[0], visual.shape[1]))
                # convert array to an image
                image = Image.fromarray(visual)
                # plot the image
                if plot_grid == 1:
                    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
                else:
                    ax[row, col].imshow(image, cmap='gray', vmin=0, vmax=255)
            
            else:
                # convert to int for matplotlib
                visual = visual.astype(int)
                # plot the image
                if plot_grid == 1:
                    ax.imshow(visual)
                else:
                    ax[row, col].imshow(visual)
            
            # remove axes
            if plot_grid == 1:
                ax.axis('off')
            else:
                ax[row, col].axis('off')
            
            i += 1

    # save the figure to file
    plt.savefig(filename)

class Train:
    def __init__(self, G, D, batchsize, latent_units):
        self.G = G
        self.D = D
        self.batchsize = batchsize
        self.latent_units = latent_units
        self.history = pd.DataFrame({
            'gen_loss': [],
            'disc_loss': []
        })


    def step(self, sequences):
        # create noise
        if type(self.latent_units) is list:
            # this means we are using the cycleGAN generator, so we feed in
            # an 'image' of noise
            noise = tf.random.normal(
                    [self.batchsize, self.latent_units[0], self.latent_units[1], 1]
                    )
        else:
            noise = tf.random.normal([self.batchsize, self.latent_units])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # generate fake sequences with the generator model
            generated_sequences = self.G.model(noise, training=True)

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

    def fit(self, dataset, epochs, vis=False, plot_grid=4, colour=False,
            notify_epoch=100, checkpoint=1000, vis_freq=1, figsize=None,
            vis_arrays=None):
        # mkdir if it does not exist
        if not os.path.exists(f'../visuals/{now}'):
            os.mkdir(f'../visuals/{now}')
            
        # so we have the same noise profile throughout
        gen_noise = tf.random.normal([self.batchsize, self.latent_units])

        # check how many data batches we have
        num_batches = 0
        for batch in dataset:
            num_batches += 1

        for e in range(epochs):
            # for every epoch, choose 'vis_freq' random steps to visualise
            step_idx = np.random.randint(0, num_batches, vis_freq)
            print(f"Epoch {e}")  # so we now the epoch number

            # run a step
            for j, sequences in enumerate(dataset):
                self.step(sequences)

                if j in step_idx:
                    if vis_arrays is None:
                        # now we visualise the results of this step, saving to file
                        # generate image
                        generated_sequence = self.G.model(gen_noise, training=False)
                        # create gen save filename
                        gen_save = f'../visuals/{now}/epoch_{e}_iter_{j}.jpg'
                        # save image
                        save_gen_image(
                            [1],
                            plot_grid,
                            gen_save,
                            self.sequences['generated'],
                            colour,
                            figsize
                        )
                    else:
                        for a, arr in enumerate(vis_arrays):
                            # reshape array
                            arr = tf.reshape(arr,
                                             (1, self.latent_units[0], self.latent_units[1], 1))
                            # generate images
                            generated_sequence = self.G.model(arr)
                            # create gen save filename
                            gen_save = f'../visuals/{now}/epoch_{e}_img_{a}.jpg'
                            # save image
                            save_gen_image(
                                [0],
                                1,
                                gen_save,
                                generated_sequence,
                                colour,
                                figsize,
                                single=True
                            )
                    
            if e % notify_epoch == 0:
                txt = ("Synthwave GAN update as of "
                       f"{datetime.now().strftime('%H:%M:%S')}.\n\n")
                plt.clf()
                fig = plt.figure(figsize=(18, 12))

                sns.lineplot(x=range(len(self.history)), y=self.history['gen_loss'],
                                        label='G', color='#5AFFE7')
                sns.lineplot(x=range(len(self.history)), y=self.history['disc_loss'],
                                        label='D', color='#726EFF')
                plt.savefig(f'../visuals/{now}/epoch_{e}_loss.jpg')
                
                
                # initialise matplotlib object for generated images
                fig, ax = plt.subplots(plot_grid, plot_grid,
                                       figsize=(plot_grid*8, plot_grid*8))
                fig.set_tight_layout(True)
                                        
                msg = notify.message(
                    subject='Synthwave GAN',
                    text=txt,
                    img=[
                        gen_save,
                        f'../visuals/{now}/epoch_{e}_loss.jpg'
                    ]
                )
                
                
                notify.send(msg)

            print(f"gen_loss: {self.history['gen_loss'].iloc[len(self.history)-1]}\ndisc_loss: {self.history['disc_loss'].iloc[len(self.history)-1]}")

            if e % checkpoint == 0:
                self.save(f"GAN{now}_{e}")

            if vis:
                plt.clf()
                fig = plt.figure(figsize=(18, 12))

                sns.lineplot(x=range(len(self.history)), y=self.history['gen_loss'],
                                        label='G', color='#5AFFE7')
                sns.lineplot(x=range(len(self.history)), y=self.history['disc_loss'],
                                        label='D', color='#726EFF')
                plt.show()

    #def show(self, array):
    #    fig = plt.figure(figsize=(18, 12))
    #    image = Image.fromarray(array)
    #    plt.imshow(image)
    #    plt.show()
        
    def save(self, modelname="gan"):
        # check the correct directories exist
        if not os.path.exists('../models'):
            os.mkdir('../models')
        if not os.path.exists('../models/tmp'):
            os.mkdir('../models/tmp')

        # save model weights
        self.G.model.save_weights(os.path.join('../models/tmp', modelname))
        # update the user
        print(f"Model '{modelname}' saved to file.")
