import os

# from  matplotlib import pyplot as plt
# import matplotlib.image as mpimg
import numpy as np

import cv2
import tensorflow as tf
from tensorflow.keras import layers, models


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_dataset(img_folder, width=32, height=32):
  img_data_array=[]
  class_name=[]
  
  for dir1 in os.listdir(img_folder):
      for file in os.listdir(os.path.join(img_folder, dir1)):
      
          image_path= os.path.join(img_folder, dir1,  file)
          image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
          image=cv2.resize(image, (height, width),interpolation = cv2.INTER_AREA)
          image=np.array(image)
          image = image.astype('float32')
          image /= 255 
          img_data_array.append(image)
          class_name.append(dir1)
  return np.array(img_data_array), np.array(class_name)


def load_dataset(trainSnow=False):
  img_folder = f'{ROOT_DIR}/../data/originalcp'
  if trainSnow:
    img_folder = f'{ROOT_DIR}/../data/ml-examples'

  img_data, class_names = create_dataset(img_folder)

  return img_data, class_names


def train_model():
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))

  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(43))

  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  
  return model


def fit():  
  img_data, class_names = load_dataset()
  target_dict = {k: v for v, k in enumerate(np.unique(class_names))}
  target_val = [target_dict[class_names[i]] for i in range(len(class_names))]
  shuffled_img_data = tf.random.shuffle(img_data, seed=12345)
  shuffled_labels = tf.random.shuffle(target_val, seed=12345)

  model = train_model()
  history = model.fit(x=np.array(shuffled_img_data, np.float32), y=np.array(shuffled_labels, np.float32), epochs=5)


fit()