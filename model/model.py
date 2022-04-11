import os

from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

import model_utils as mu
from model_utils import DATA_DIR


def load_dataset(trainSnow=True):
    if trainSnow:
        img_folder, ext = os.path.join(DATA_DIR, 'ml-examples'), '.png'
    else:
        img_folder, ext = os.path.join(DATA_DIR, 'original'), '.ppm'

    img_data, class_names = mu.load_images(img_folder, ext)

    normalized_classes = mu.normalize_labels(class_names)

    return img_data, normalized_classes


def init_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model


def init_and_fit():
    img_data, class_names = load_dataset()
    model = init_model()
    history = model.fit(x=np.array(img_data, np.float32), y=np.array(list(class_names), np.float32), epochs=3)

if __name__ == "__main__":
    init_and_fit()
