import os

from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models

import model_utils as mu
from model_utils import DATA_DIR

# what proportion of data is used as training data
TRAIN_RATIO = 0.8

def load_dataset(trainSnow=True):
    if trainSnow:
        img_folder, ext = os.path.join(DATA_DIR, 'ml-examples'), '.png'
    else:
        img_folder, ext = os.path.join(DATA_DIR, 'original'), '.ppm'

    img_data, class_names = mu.load_images(img_folder, ext)

    normalized_classes = mu.normalize_labels(class_names)

    return img_data, normalized_classes


def init_model(num_classes):
    """"
    returns an untrained classifier model
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model


if __name__ == "__main__":
    img_data, class_names = load_dataset(trainSnow=True)
    num_train = int(TRAIN_RATIO * len(img_data))

    img_array = mu.wrap_np_array(img_data)
    train_img, test_img = img_array[:num_train], img_array[num_train:]

    class_array = mu.wrap_np_array(list(class_names))
    train_label, test_label = class_array[:num_train], class_array[num_train:]
    
    num_classes = len(set(class_names))

    init_model(num_classes=num_classes).fit(
        x = train_img,
        y = train_label,
        epochs=6,
        validation_data = (test_img, test_label),
        shuffle=True
        )
    
