import cv2
import numpy as np


def resize_image(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    """Re-sizes an image

    An image along with the width and height to be resized can be passed in
    and the function will return a resized version of the image.
    """
    # get image width and height
    (_height, _width) = image.shape[:2]

    # check if the image is
    if width is None and height is None:
        return image

    # if the width is None
    if width is None:
        ratio = height / float(_height)
        dimensions = (int(_width * ratio), height)
    elif height is None:
        ratio = width / float(_width)
        dimensions = (width, int(_height * ratio))
    else:
        dimensions = None

    # resize the image
    return cv2.resize(image, dimensions, interpolation=interpolation)


def crop_image(image, start_y, end_y, start_x, end_x):
    return image[start_y:end_y, start_x:end_x]


def split_image_in_half(image, direction):
    part1, part2 = None, None
    (height, width) = image.shape[:2]
    if direction == "vertical":
        part1 = image[0:int(height/2), 0:width]
        part2 = image[int(height/2):height, 0: width]
    elif direction == 'horizontal':
        part1 = image[0:height, 0:int(width/2)]
        part2 = image[0:height, int(width/2):width]

    return np.array([part1, part2])


def convert_to_rgb(image):
    return image.convert("RGB")


def prep_image(image, to_array=True):
    # check if image is a numpy array
    if isinstance(image, np.ndarray):
        return image
    # check if image is RGB, if not convert
    if image.mode != "RGB":
        image = convert_to_rgb(image)

    # if to array is set to true, convert to a numpy array
    if to_array:
        image = np.array(image)

    return image
