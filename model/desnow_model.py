from importlib.metadata import PathDistribution
import os
from pickletools import optimize

from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random

import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from model import create_dataset, load_dataset

# Code based on `https://towardsdatascience.com/image-noise-reduction-in-10-minutes-with-convolutional-autoencoders-d16219d2956a`

__imageX = 32
__imageY = 32

class NoiseReducer(tf.keras.Model):
    def __init__(self) -> None:
        super(NoiseReducer, self).__init__()

        self.encoder = tf.keras.Sequential([
          layers.Input(shape=(__imageX, __imageY, 1)),
          layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
          layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
          layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
          layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
          layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# returns a trained model
def train_model(clean_train, noisy_train, clean_test, noisy_test):
  autoencoder = NoiseReducer()

  autoencoder.compile(optimizer='adam', loss='mse')
  autoencoder.fit(noisy_train,
                  clean_train,
                  epochs=10,
                  shuffle=True,
                  validation_data=(noisy_test, clean_test))

  return autoencoder


def load_dataset_and_train_model():
  noisy_img_data, class_names = load_dataset(trainSnow=True)
  clean_img_data, class_names = load_dataset(trainSnow=False)

  num_noisy_total = len(noisy_img_data)
  num_noisy_train = 4 * num_noisy_total // 5

  num_clean_total = len(clean_img_data)
  num_clean_train = 4 * num_clean_total // 5

  model = train_model(
    clean_img_data[:num_clean_train],
    noisy_img_data[:num_noisy_train],
    clean_test= clean_img_data[num_clean_train:],
    noisy_test= noisy_img_data[num_noisy_train:])

if __name__ == "__main__":
  
  # imgg = cv2.imread(os.path.join(os.path.dirname(__file__), 'testimage.ppm'), cv2.COLOR_BGR2RGB)
  # cv2.imshow('window', imgg)
  # cv2.waitKey(5000)
  # cv2.imwrite('testimage.png', imgg)
  load_dataset_and_train_model()
