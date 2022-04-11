import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# code based on https://www.tensorflow.org/tutorials/generative/autoencoder

class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
        layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
        layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
        layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def print_summaries(self):
        self.encoder.summary()
        self.decoder.summary()


def trained_model(x_train, x_test, x_train_noisy, x_test_noisy, num_epochs=10):
    """Returns an autoencoder trained on the given data"""
    autoencoder = Denoise()

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.fit(x_train_noisy, x_train,
                    epochs=num_epochs,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))
    return autoencoder


def plot_input_vs_output(x_test_noisy, decoded_imgs, n):
    plt.figure(figsize=(20, 4))
    for i in range(n):

        # display original + noise
        ax = plt.subplot(2, n, i + 1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(x_test_noisy[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

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
    (x_train, _), (x_test, _) = fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # noisy
    noise_factor = 0.2
    x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
    x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape) 

    x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
    x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

    # train model on data
    denoiser_model = trained_model(x_train, x_test, x_train_noisy, x_test_noisy, 3)

    # use model to denoise noisy images
    decoded_imgs = denoiser_model(x_test_noisy)

    # plot them
    plot_input_vs_output(x_test_noisy, decoded_imgs, n=10)

