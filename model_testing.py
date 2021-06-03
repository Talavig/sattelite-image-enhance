import tensorflow as tf

import time
import os
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint

from cv2 import imread, imshow, waitKey

from datatools import get_data_from_folder

max_batches = 2500
batch_size = 64
epochs = 3

train_dir = 'C:\\project_satellite_pycharm\\pictures\\train\\'
dev_dir = 'C:\\project_satellite_pycharm\\pictures\\dev\\'
test_dir = 'C:\\project_satellite_pycharm\\pictures\\test\\'

model_name = 'C:\\project_satellite_pycharm\\satellite_image_model'


def get_model():
    """

    :return: get the keras model
    """
    model_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    #arguments which all convolution layers use
    inputs = keras.Input(shape=(256, 256, 3))
    x = layers.Conv2D(128, 5, **model_args)(inputs)
    x = layers.Conv2D(64, 5, **model_args)(x)
    x = layers.Conv2D(64, 4, **model_args)(x)
    x = layers.Conv2D(32, 4, **model_args)(x)
    x = layers.Conv2D(32, 3, **model_args)(x)
    x = layers.Conv2D(16, 3, **model_args)(x)
    x = layers.Conv2D(8, 3, **model_args)(x)
    outputs = layers.Conv2DTranspose(3, 5, **model_args)(x)

    return keras.Model(inputs, outputs)


def calculate_img_stats(real_img, img_to_compare):
    """
    :param real_img: the regular, unmanipulated image
    :param img_to_compare: the image we compare to the original
    :return: print image_to_compare stats relative to the original
    """
    print('psnr: ' + str(tf.image.psnr(real_img, img_to_compare, max_val=255)))
    print('ssim: ' + str(tf.image.ssim(real_img, img_to_compare, max_val=255)))


def generate_samples(input, regular, batch_size):
    """

    :param input: os.scandir object, represents the input (blurred) images
    :param regular: os.scandir object, represents the corresponding regular (unblurred) images
    :param batch_size: the amount of images in the batch
    :return: generate the samples for learning
    """
    while True:
        batch_x = []
        batch_y = []
        counter = 0
        for x, y in zip(input, regular):
            if counter <= batch_size:
                batch_x.append(imread(x.path))
                batch_y.append(imread(y.path))
                counter += 1
            else:
                break
        yield np.asarray(batch_x), np.asarray(batch_y)
        batch_x = []
        batch_y = []


def predict_unblurred_img(img_to_predict, model):
    """

    :param img_to_predict: the blurred image we want to test the model on
    :param model: the trained keras model
    :return: an unblurred image, as predicted by the model
    """
    img_to_predict = np.reshape(img_to_predict, (1, 256, 256, 3))
    prediction = np.reshape(np.round(model.predict(img_to_predict)), (256, 256, 3)).astype('uint8')
    return prediction


def show_model_info(prediction, real, to_predict):
    """

       :param prediction: the output of the model
       :param real: the actual image
       :param to_predict: the input to the model
       :return: show stats about the model preformance
       """
    print(prediction)
    print(real)
    imshow('prediction', prediction)
    waitKey(0)

    print('original:')
    calculate_img_stats(real, to_predict)
    print('\n')
    print('prediction:')
    calculate_img_stats(real, tf.cast(prediction, tf.uint8))


def main():
    t = time.time()
    train_input = os.scandir(train_dir + 'input')
    train_regular = os.scandir(train_dir + 'regular')

    checkpoint = ModelCheckpoint("C:\\project_satellite_pycharm\\checkpoint", monitor='accuracy', verbose=1,
                                 save_best_only=True, mode='auto', save_freq='epoch')

    (x_test, y_test) = get_data_from_folder(test_dir, 5000)

    model = get_model()

    model.compile(
        loss='mean_absolute_error',
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    history = model.fit(generate_samples(train_input, train_regular, batch_size), steps_per_epoch=max_batches,
                        epochs=epochs, callbacks=[checkpoint])

    test_scores = model.evaluate(x_test, y_test, verbose=2)

    x_test = []
    y_test = []

    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    model.save(model_name)

    # model = tf.keras.models.load_model('C:\\project_satellite_pycharm\\satellite_image_model')
    # model.summary()
    img_to_predict = imread(train_dir + 'input\\img0.jpg')
    img_real = imread(train_dir + 'regular\\img0.jpg')

    prediction = predict_unblurred_img(img_to_predict, model)
    show_model_info(prediction, img_real, img_to_predict)

    print(time.time() - t)


if __name__ == '__main__':
    main()
