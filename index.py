from flask import Flask, render_template, flash, request, redirect, url_for, make_response
import web_utils
import cv2
import numpy as np
import filetype

app = Flask(__name__)


def get_img_from_binary(image_bin):
    """

    :param image_bin: the binary content of the imgae as recieved from the user
    :return: the image as a cv2 image
    """
    npimg = np.fromstring(image_bin, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img


def check_input_file(file_data):
    """

    :param file_data: the file binary
    :return: True if file is an image of any format (jpg,png,tiff ect.), False otherwise
    """
    return filetype.is_image(file_data)


def get_best_images(img):
    """

    :param img: the image binary
    :return: a list of the best changed sections as specified in web_utils (currently returns one section, aka the entire picture)
    """
    img = get_img_from_binary(img)
    return web_utils.get_best_diff(img)


@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":

        if request.files:
            image = request.files["image"]

            img_data = image.read()

            if check_input_file(img_data):
                best_diff_images = get_best_images(img_data)
                response = make_response(render_template("upload_image.html", show_results=True))

                return add_header(response)

    return render_template("upload_image.html", show_results=False)


@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/test")
def test():
    return render_template("test.html")


if __name__ == '__main__':
    app.run(debug=True)
