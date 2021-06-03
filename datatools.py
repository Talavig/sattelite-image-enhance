
import os
from cv2 import imread
import threading
from multiprocessing.pool import ThreadPool, Pool
from natsort import natsorted
import numpy as np

import tensorflow as tf

train_dir = 'C:\\project_satellite_pycharm\\pictures\\train\\'
dev_dir = 'C:\\project_satellite_pycharm\\pictures\\dev\\'
test_dir = 'C:\\project_satellite_pycharm\\pictures\\test\\'


def remove_empties(data_dir):
    """

    :param data_dir: folder from which we want to remove empty files
    :return: removers all empty files from folder.
    """
    for file in os.scandir(data_dir):
        try:
            if os.path.getsize(file.path) == 0:
                os.remove(file.path)
                print(file.path)
        except:
            continue


def remove_singles(data_dir):
    """

        :param data_dir: folder from which we want to remove single filed
        :return: removers all files where unblurred version exists and an unblurred one does not.
        """
    for file in os.scandir(data_dir):
        try:
            if os.path.exists(file.path) and not os.path.exists(file.path):
                os.remove(file.path)
                print(file.path)
        except:
            continue


def get_list_img(data_dir, chunk_size):
    """

    :param data_dir: the location of the folder where the images are stored
    :param chunk_size:the amout of images needed
    :return:an array of np arrays representing the images
    """
    img_locations = natsorted([file.path for file in os.scandir(data_dir)])[0:chunk_size]
    return np.asarray([imread(file) for file in img_locations])


def get_data_from_folder(data_dir, chunk_size):
    """

        :param data_dir: the location of the folder where the images are stored
        :param chunk_size:the amout of images needed
        :return:an array of np arrays representing the images, with correnspondence between blurred and unblurred images
        """
    x_dir = data_dir + 'input\\'
    y_dir = data_dir + 'regular\\'

    pool = ThreadPool(processes=4)
    tx = pool.apply_async(get_list_img, (x_dir, chunk_size))
    ty = pool.apply_async(get_list_img, (y_dir, chunk_size))
    return tx.get(), ty.get()


def sort_images_into_classes(data_dir):
    """

    :param data_dir: the location of the folder where the images are stored
    :return: sort images into "input"(blurred) and "regular"(unblurred) folders based on name
    """

    def input(img_locations):
        counter = 0
        for file in img_locations:
            os.rename(data_dir + file, data_dir + 'input\\img' + str(counter) + '.jpg')
            counter += 1

    def regular(img_locations):
        counter = 0
        for file in img_locations:
            os.rename(data_dir + file, data_dir + 'regular\\img' + str(counter) + '.jpg')
            counter += 1

    imgs = natsorted(os.listdir(data_dir))
    t1 = threading.Thread(target=input, args=([imgs[1:][::2]]))
    t2 = threading.Thread(target=regular, args=([imgs[0:][::2]]))
    t1.start()
    t2.start()


def calculate_img_stats(real_img, img_to_compare):
    """

    :param real_img: the regular, unmanipulated image
    :param img_to_compare: the image we compare to the original
    :return: print image_to_compare stats relative to the original
    """
    print('psnr: ' + str(tf.image.psnr(real_img, img_to_compare, max_val=255)))
    print('ssim: ' + str(tf.image.ssim(real_img, img_to_compare, max_val=255)))


def show_model_info(prediction, real, to_predict):
    """

    :param prediction: the output of the model
    :param real: the actual image
    :param to_predict: the input to the model
    :return: show stats about the model preformance
    """
    print(prediction)
    print(real)

    print('original:')
    calculate_img_stats(real, to_predict)
    print('\n')
    print('prediction:')
    calculate_img_stats(real, tf.cast(prediction, tf.uint8))


def main():


if __name__ == '__main__':
    main()
