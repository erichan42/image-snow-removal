import os

# import modules
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

import desnow_model as ds
import model_utils as mu
import save_models as sm


CHECKPOINTS = './../model/save'
DATA = './../data'


noisy_imgs, noisy_labels = mu.load_images(
        img_dir=os.path.join(DATA, 'ml-examples'),
        img_ext='.png',
        )

denoiser = ds.Denoise()
denoiser.load_weights(f'{CHECKPOINTS}/denoiser')
loss, acc = denoiser.evaluate(noisy_imgs[0], noisy_labels[0], verbose=2)
