import os

import sklearn.model_selection as sk
import tensorflow as tf

import desnow_model as ds
import classifier_model as cl
import model_utils as mu


TEST_RATIO = 0.2
SAVE_DIR = 'save'


def train_desnow(clean, noisy, epochs=5):
    clean_train, clean_test, noisy_train, noisy_test = sk.train_test_split(clean, noisy, test_size=TEST_RATIO, random_state = 42)

    # train model on data
    return ds.trained_denoiser(
                clean_train,
                clean_test,
                noisy_train,
                noisy_test,
                num_epochs=epochs)


def train_classifier(img_array, class_array, num_classes, epochs=5):
    train_img, test_img, train_label, test_label = sk.train_test_split(img_array, class_array, test_size=TEST_RATIO, random_state = 42)
    
    cl_model = cl.init_model(num_classes=num_classes)
    cl_model.fit(
        x = train_img,
        y = train_label,
        epochs=epochs,
        shuffle=True,
        validation_data = (test_img, test_label)
        )

    return cl_model


if __name__ == '__main__':
    clean_imgs, clean_labels = mu.load_corresponding_images(
        pattern_dir=os.path.join(mu.DATA_DIR, 'ml-examples'),
        src_dir=os.path.join(mu.DATA_DIR, 'original'),
        pattern_ext='.png',
        src_ext='.ppm',
        )
    clean_imgs = mu.wrap_np_array(clean_imgs)
    clean_labels = mu.wrap_np_array(clean_labels)

    noisy_imgs, noisy_labels = mu.load_images(
        img_dir=os.path.join(mu.DATA_DIR, 'ml-examples'),
        img_ext='.png',
        )
    noisy_imgs = mu.wrap_np_array(noisy_imgs)
    noisy_labels = mu.wrap_np_array(noisy_labels)


    denoiser = train_desnow(clean_imgs, noisy_imgs)
    denoiser.save_weights(f'./{SAVE_DIR}/denoiser')

    classifier_orig = train_classifier(clean_imgs, clean_labels, len(set(clean_labels)))
    classifier_orig.save_weights(f'./{SAVE_DIR}/classifier_orig')

    classifier_noisy = train_classifier(noisy_imgs, noisy_labels, len(set(clean_labels)))
    classifier_noisy.save_weights(f'./{SAVE_DIR}/classifier_noisy')
