from typing import Tuple, Any

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop


class Generator(Model):
    def __init__(self, in_dim: int, out_channels: int):
        super().__init__()
        self.sequential = Sequential(name='Generator')
        self.sequential.add(Dense(128 * 8 * 8, input_dim=in_dim, activation="relu"))  # 100
        self.sequential.add(Reshape((8, 8, 128)))  # 128 * 8 * 8
        self.sequential.add(UpSampling2D())  # 8x8
        self.sequential.add(Conv2D(128, kernel_size=4, padding="same"))  # 16x16
        self.sequential.add(BatchNormalization(momentum=0.8))
        self.sequential.add(Activation("relu"))
        self.sequential.add(UpSampling2D())  # 16x16
        self.sequential.add(Conv2D(64, kernel_size=4, padding="same"))  # 32x32
        self.sequential.add(BatchNormalization(momentum=0.8))
        self.sequential.add(Activation("relu"))
        self.sequential.add(Conv2D(out_channels, kernel_size=4, padding="same"))
        self.sequential.add(Activation("tanh"))

    def call(self, inputs, **kwargs):
        return self.sequential(inputs)

    def summary(self, *args, **kwargs):
        self.sequential.summary(*args, **kwargs)


class Discriminator(Model):
    def __init__(self, in_shape: Tuple[int, int, int]):
        super().__init__()
        self.sequential = Sequential()

        self.sequential.add(Conv2D(16, kernel_size=3, input_shape=in_shape,
                                   strides=2, padding="same"))
        self.sequential.add(LeakyReLU(alpha=0.2))
        self.sequential.add(Dropout(0.25))
        self.sequential.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        self.sequential.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        self.sequential.add(BatchNormalization(momentum=0.8))
        self.sequential.add(LeakyReLU(alpha=0.2))
        self.sequential.add(Dropout(0.25))
        self.sequential.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        self.sequential.add(BatchNormalization(momentum=0.8))
        self.sequential.add(LeakyReLU(alpha=0.2))
        self.sequential.add(Dropout(0.25))
        self.sequential.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        self.sequential.add(BatchNormalization(momentum=0.8))
        self.sequential.add(LeakyReLU(alpha=0.2))
        self.sequential.add(Dropout(0.25))

        self.sequential.add(Flatten())
        self.sequential.add(Dense(1))

    def call(self, inputs, **kwargs):
        return self.sequential(inputs)

    def summary(self, *args, **kwargs):
        self.sequential.summary(*args, **kwargs)


class WGAN(Model):
    def __init__(self,
                 generator: Model,
                 discriminator: Model,
                 latent_dim: int,
                 ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn, **kwargs):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        generated_images = self.generator(random_latent_vectors)

        combined_images = tf.concat([generated_images, real_images], axis=0)

        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}
