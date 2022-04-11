import os

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import model_utils as mu

# code based on https://www.tensorflow.org/tutorials/generative/autoencoder

class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
        layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
        layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
        layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def print_summaries(self):
        self.encoder.summary()
        self.decoder.summary()


def trained_denoiser(x_train, x_test, x_train_noisy, x_test_noisy, num_epochs=10):
    """Returns an autoencoder trained on the given data"""
    autoencoder = Denoise()

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.fit(x_train_noisy, x_train,
                    epochs=num_epochs,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))
    return autoencoder


def plot_input_vs_output(x_test_clean, x_test_noisy, decoded_imgs, n):
    plt.figure(figsize=(20, 4))
    for i in range(n):

        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.title("original")
        plt.imshow(tf.squeeze(x_test_clean[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # # display original + noise
        # ax = plt.subplot(2, n, i + 1)
        # plt.title("original + noise")
        # plt.imshow(tf.squeeze(x_test_noisy[i]))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

        # display reconstruction
        bx = plt.subplot(2, n, i + n + 1)
        plt.title("reconstructed")
        plt.imshow(tf.squeeze(decoded_imgs[i]))
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)
    plt.show()


# load snow data, train model, decode sample, display
if __name__ == "__main__":
    
    # clean
    clean_imgs, clean_labels = mu.load_corresponding_images(
        pattern_dir=os.path.join(mu.DATA_DIR, 'ml-examples'),
        src_dir=os.path.join(mu.DATA_DIR, 'original'),
        pattern_ext='.png',
        src_ext='.ppm',
        )

    num_train = int(0.8 * len(clean_imgs))

    clean_img_array = mu.wrap_np_array(clean_imgs)[..., tf.newaxis]
    clean_train = clean_img_array[:num_train] # first n
    clean_test = clean_img_array[num_train:] # remainder
    
    # noisy
    noisy_imgs, noisy_labels = mu.load_images(
        img_dir=os.path.join(mu.DATA_DIR, 'ml-examples'),
        img_ext='.png',
        )
    noisy_img_array = mu.wrap_np_array(noisy_imgs)[..., tf.newaxis]
    noisy_train = noisy_img_array[:num_train] # first n
    noisy_test = noisy_img_array[num_train:] # remainder

    # train model on data
    denoiser_model = trained_denoiser(clean_train, clean_test, noisy_train, noisy_test, num_epochs=5)

    # use model to denoise noisy images
    denoised_imgs = denoiser_model(noisy_test)

    # plot them
    plot_input_vs_output(clean_test, noisy_test, denoised_imgs, n=10)

