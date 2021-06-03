import requests
import random
from PIL import Image
import threading
import os


save_to_dir = 'C:\\project_satellite_pycharm\\pictures\\train\\'
train_dir = 'C:\\project_satellite_pycharm\\pictures\\train\\'
dev_dir = 'C:\\project_satellite_pycharm\\pictures\\dev\\'
test_dir = 'C:\\project_satellite_pycharm\\pictures\\test\\'

SIZE = 16

# min row, min column, max row, max column
MIN_ROW = 0
MIN_COLUMN = 1
MAX_ROW = 2
MAX_COLUMN = 3

# note: img num and jump must divide eachother perfectly, img num creates twice as many pictures (blurred and regular)
IMG_NUM = 500000
JUMP = 50

random_location = [10000, 10000, 99999, 99999]

cities = {
    'barcelona': [24457, 24497, 33122, 33180],
    'london': [21712, 32473, 22011, 33003],
    'stockholm': [19209, 35946, 19342, 36118],
    'buddapesht': [22879, 36203, 22961, 36311],
    'mosco': [20433, 39547, 20606, 39684],
    'tokyo': [25626, 58046, 25926, 58309],
    'beijing': [24747, 53850, 24941, 54083],
    'shanghai': [26555, 54507, 26938, 54964],
    'san_francisco': [25235, 10441, 25467, 10603],
    'rio': [36992, 24806, 37063, 24986],
    'sao_paulo': [37156, 24219, 37232, 24353],
    'new_york': [24559, 19185, 24661, 19428],
    'paris': [22482, 33087, 22617, 33260],
    'mombai': [29151, 46010, 29268, 46053],
    'new_delhi': [27280, 46769, 27394, 46892],
    'mexico_city': [29114, 14682, 29204, 14763],
    'toronto': [23846, 18213, 24028, 18367],
    'brazilia': [35661, 23993, 35714, 24069],
    'rome': [24332, 35013, 24382, 35067],
    'athene': [25251, 37041, 25322, 37110],
    'vegas': [25657, 11767, 25740, 11838],
    'chicago': [24286, 16706, 24469, 16874],
    'milan': [23373, 34344, 23488, 34535],
    'dalas': [26346, 15024, 26545, 15227],
    'denver': [24792, 13598, 24920, 13704],
    'cairo': [26991, 38412, 27089, 38495],
    'albaqurki': [25894, 13330, 25969, 13383],
    'sydney': [39274, 60190, 39377, 60299],
    'madrid': [24680, 32048, 24760, 32131],
    'warsaw': [21540, 36554, 21633, 36635],
    'istanbul': [24540, 37964, 24589, 38069],
    'accra': [31730, 32673, 31776, 32765],
    'johannesburg': [37652, 37832, 37763, 37959],
    'bagdad': [26304, 40817, 26354, 40871],
    'dubai': [27965, 42744, 28102, 42908],
    'bangkok': [30172, 50995, 30309, 51160],
    'singapore': [32459, 51605, 32582, 51728],
    'jacarata': [33862, 52134, 33983, 52278],
    'hong_kong': [28338, 53223, 28608, 53615],
    'vanccover': [22415, 10325, 22484, 10413],
    'bogota': [31894, 19249, 31950, 19294],
    'buenos_aires': [39420, 22044, 39595, 22219],
    'melbourne': [40162, 59086, 40381, 59244]
}

BASE_URL = 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/'
BAD_IMAGE = open('bad_image.jpg', 'rb').read()
MIN_FILE_SIZE = 5000
ORIGINAL_NAME_END = -4
COMPRESSION_RATE = 72

IMG_SIZES_LIST = [0.2, 0.25, 0.4, 0.5, 0.625]
IMG_SIZE_TUPLE = (256, 256)


def img_name_maker(img_num):
    """

    :param img_num: the number of the image for naming purposes
    :return: the full location of the image
    """
    return save_to_dir + str(img_num) + '.jpg'


def blurred_img_name_maker(img_name):
    """

    :param img_name: the full name of the original image
    :return: the full location of the blurred image
    """
    return img_name[:ORIGINAL_NAME_END] + 'blur.jpg'


def generate_url_all_world():
    """

    :return: get full url for requesting image, image from all over the world
    """
    row = str(random.randint(random_location[MIN_ROW], random_location[MAX_ROW]))
    column = str(random.randint(random_location[MIN_COLUMN], random_location[MAX_ROW]))
    return BASE_URL + str(SIZE) + '/' + str(row) + '/' + str(column)


def generate_url_city():
    """

    :return: get full url for requesting image, image from a randomly chosen city
    """
    city_coordinates = random.choice(list(cities.values()))
    row = str(random.randint(city_coordinates[MIN_ROW], city_coordinates[MAX_ROW]))
    column = str(random.randint(city_coordinates[MIN_COLUMN], city_coordinates[MAX_COLUMN]))
    return BASE_URL + str(SIZE) + '/' + str(row) + '/' + str(column)


def get_image_data_from_url(url):
    """

    :param url: the url for the http request
    :return: the content of the request, aka the image data
    """
    r = requests.get(url)
    return r.content


def create_new_image(img, html_img_data, img_name):
    """
    this function is activated if the image number does not exist already
    :param img: file object for image saving and manipulating
    :param html_img_data: the data of the file we want to save
    :param img_name: the full location of the image object
    :return: save the original and blurred image
    """
    size_small_img = random.choice(IMG_SIZES_LIST)
    new_size = tuple([int(size_small_img * x) for x in IMG_SIZE_TUPLE])
    # choose a scaling factor and create a tuple which represents the new size of the image

    img.write(html_img_data)
    img_in_pil = Image.open(img_name)

    small_image = img_in_pil.resize(new_size)
    blurred_img = small_image.resize(IMG_SIZE_TUPLE).convert('RGB')
    # enlarge and diminish image in order to recreate the blurring effect of zooming

    blurred_img.save(blurred_img_name_maker(img_name))


def create_existing_image(img_name):
    """
    this function is activated if the image number already exists without a blurred counterpart
    :param img_name: the full location of the image object
    :return: save the original and blurred image
    """
    size_small_img = random.choice(IMG_SIZES_LIST)
    new_size = tuple([int(size_small_img * x) for x in IMG_SIZE_TUPLE])
    #choose a scaling factor and create a tuple which represents the new size of the image

    img_in_pil = Image.open(img_name)

    small_image = img_in_pil.resize(new_size)
    blurred_img = small_image.resize(IMG_SIZE_TUPLE).convert('RGB')
    #enlarge and diminish image in order to recreate the blurring effect of zooming

    blurred_img.save(blurred_img_name_maker(img_name))


def image_making(img_range):
    """

    :param img_range: a tuple representing the range of image numbers to be created by the thread
    :return: saves the images
    """
    for img_num in img_range:
        #iterate over images in range of thread

        img_name = img_name_maker(img_num)
        if not os.path.exists(img_name):
            #if this image name does not exist

            with open(img_name, 'wb') as img:
                while True:

                    generating_func = random.choice([generate_url_city, generate_url_all_world])
                    url = generating_func()
                    html_img_data = get_image_data_from_url(url)

                    if html_img_data != BAD_IMAGE and len(html_img_data) > MIN_FILE_SIZE:
                        #if the image recived has meaningful content and the image exists on the grid

                        try:
                            create_new_image(img, html_img_data, img_name)
                            break
                        except:
                            continue

        elif not os.path.exists(blurred_img_name_maker(img_name)):
            #if only the regular image exists without a blurred counterpart
            try:
                create_existing_image(img_name)
            except:
                continue


def main():
    for img_num in range(0, IMG_NUM, int(IMG_NUM / JUMP)):
        img_range = range(img_num, img_num + int(IMG_NUM / JUMP))
        t = threading.Thread(target=image_making, args=(img_range,))
        t.start()


if __name__ == '__main__':
    main()
