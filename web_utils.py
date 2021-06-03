import cv2
import tensorflow as tf
import numpy as np
from skimage.metrics import structural_similarity

small_img_size = 256  # this means that we only return one image(the amount of images created is 256/small_img_size), can be 256,128 64,32,16,8
needed_img_size = 256
w = h = 128
best_count = 3
IMG_SIZE_TUPLE = (needed_img_size, needed_img_size)
img_test_loc_blur = "C:\\project_satellite_pycharm\\pictures\\dev\\input\\img2.jpg"
img_test_loc_real = "C:\\project_satellite_pycharm\\pictures\\dev\\regular\\img2.jpg"
model_name = 'C:\\project_satellite_pycharm\\satellite_image_model_11'
static_path = r'static\\img\\'


def get_normalized_image(img):
    """

    :param img: a cv2 image
    :return: a 256*256 image for the model
    """
    height, width, channels = img.shape
    if height == width:
        if height == needed_img_size:
            return img
        return cv2.resize(img, IMG_SIZE_TUPLE)
    center = [int(x / 2) for x in img.shape]
    if height > width:
        x = int(width / 2)
        y = center[0] - int(h / 2)

        crop_img = img[int(y - h):int(y + h), int(0):int(width)]
        return cv2.resize(crop_img, IMG_SIZE_TUPLE)
    else:
        x = center[1] - int(w / 2)
        y = int(height / 2)

        crop_img = img[int(0):int(height), int(x - w):int(x + w)]
        return cv2.resize(crop_img, IMG_SIZE_TUPLE)


def get_split_images(img):
    """

    :param img: cv2 image size 256*256
    :return: a list of images as divided by small img size
    """
    split_images = []
    for r in range(0, img.shape[0], small_img_size):
        for c in range(0, img.shape[1], small_img_size):
            split_images.append(img[r:r + small_img_size, c:c + small_img_size, :])
    return split_images


def get_unblurred_img(img):
    """

    :param img: cv2 image size 256*256
    :return: an unblurred image as predicted by the model
    """
    model = tf.keras.models.load_model(model_name)
    img_to_predict = np.reshape(img, (1, 256, 256, 3))
    prediction = np.reshape(np.round(model.predict(img_to_predict)), (256, 256, 3)).astype('uint8')
    return prediction


def get_best_diff_list(split_images_blurred, split_images_unblurred):
    """

    :param split_images_blurred: list of blurred images as splitted from original
    :param split_images_unblurred: list of blurred images as splitted from prediction
    :return: the top (best_count) images with their corresponding unblurred counterpart based on how different they are
    """
    diff_list = []
    for blur, unblur in zip(split_images_blurred, split_images_unblurred):
        img_diff = structural_similarity(blur, unblur, multichannel=True)
        diff_list.append(([blur, unblur], img_diff))
    diff_list = sorted(diff_list, key=lambda x: x[1])
    top = []
    for item in diff_list:
        top.append(item[0])
    return top[:best_count]


def save_images_in_static(img_list):
    """

    :param img_list: the images to save
    :return: save the images in static for the ckient to load
    """
    for img_pair in enumerate(img_list):
        # place in enumerate, the image itself
        img_original = img_pair[1][0]
        img_unblured = img_pair[1][1]
        img_num = str(img_pair[0])

        cv2.imwrite(static_path + 'input' + img_num + '.jpg', img_original)
        cv2.imwrite(static_path + 'regular' + img_num + '.jpg', img_unblured)

        canny_blurred, canny_unblurred = get_edges_canny(img_original, img_unblured)

        cv2.imwrite(static_path + 'input_canny' + img_num + '.jpg', img_original)
        cv2.imwrite(static_path + 'regular_canny' + img_num + '.jpg', img_unblured)


def get_best_diff(img):
    """

    :param img: the cv2 image recieved from the client
    :return: the best pair of blurred and unblurred images as splitted
    """
    img = get_normalized_image(img)
    split_images_blurred = get_split_images(img)
    unblurred_img = get_unblurred_img(get_unblurred_img(img))
    split_images_unblurred = get_split_images(unblurred_img)
    best_diff_list = get_best_diff_list(split_images_blurred, split_images_unblurred)
    save_images_in_static(best_diff_list)
    return best_diff_list


def get_edges_canny(blurred, unblurred):
    """

    :param blurred: blurred cv2 image
    :param unblurred: unblurred cv2 image
    :return: tthe image with canny algorith mask on top
    """
    edges_blurred = cv2.Canny(blurred, 100, 200)
    edges_unblurred = cv2.Canny(unblurred, 100, 200)

    blurred[edges_blurred == 255] = [255, 0, 0]
    unblurred[edges_unblurred == 255] = [255, 0, 0]

    return blurred, unblurred


def main():
    model = tf.keras.models.load_model(model_name)
    model.summary()

if __name__ == '__main__':
    main()