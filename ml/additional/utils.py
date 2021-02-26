import errno
import os
from PIL import Image
import numpy as np


class DictAttr(dict):
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def check_and_create_dir(full_path):
    """Checks if a given path exists and if not, creates the needed directories.
            Inputs:
                full_path: path to be checked
    """
    if not os.path.exists(os.path.dirname(full_path)):
        try:
            os.makedirs(os.path.dirname(full_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def read_images_from_path(image_names):
    """ Takes in a path and a list of image file names to be loaded and returns a list of all loaded images after resize.
           Inputs:
                image_names: list of image names
           Returns:
                List of all loaded and resized images
    """
    returnValue = []
    for image_name in image_names:
        im = Image.open(image_name)
        returnIm = process_image_to_np_array(im)
        returnValue.append(returnIm)
    return returnValue


def process_image_to_np_array(image):
    imArr = np.asarray(image)

    # Remove alpha channel if exists
    if len(imArr.shape) == 3 and imArr.shape[2] == 4:
        if (np.all(imArr[:, :, 3] == imArr[0, 0, 3])):
            imArr = imArr[:, :, 0:3]
    if len(imArr.shape) != 3 or imArr.shape[2] != 3:
        raise Exception('Error: Image is not RGB.')

    return np.asarray(imArr)
