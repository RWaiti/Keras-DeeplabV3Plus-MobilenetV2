# Path: utils\image_utils.py
import os
import cv2
import numpy as np
from glob import glob
from random import shuffle, randint
from tensorflow import cast, float32, reshape, Tensor
from tensorflow.image import stateless_random_crop, flip_left_right, resize, flip_up_down

CROP_PERCENT = .75



def common_ops(img, size, CROP=None, BRIGHTNESS=False, FLIP_HORIZONTAL=False,
               FLIP_VERTICAL=False, interpolation="bilinear"):
    height, width, n_channels = img.shape

    if CROP != None:
        img = img[CROP[0][0]:CROP[0][1], CROP[1][0]:CROP[1][1], :]

    if img.shape != (size[0], size[1], 3):
        img = resize(img, size, interpolation)

    if FLIP_HORIZONTAL:
        img = flip_left_right(img)

    if FLIP_VERTICAL:
        img = flip_up_down(img)

    if isinstance(img, Tensor):
        img = img.numpy()

    if BRIGHTNESS:
        if randint(0, 1):
            img = img + 32
            img[img > 255] = 255
        else:
            img = img - 32
            img[img < 0] = 0

    return img.astype(np.uint8)


def load_mask(path, size, n_classes=2, map_func=None, CROP=None, FLIP_HORIZONTAL=False,
              FLIP_VERTICAL=False):
    """ returns a mask with shape (size, size, n_classes)

    Args:
        path (str): path to the mask
        size (tuple, optional): size of the image.
        n_classes (int, optional): number of classes. Defaults to 19.
        map_func (function, optional): function to map the mask. Defaults to None.

    Returns:
        np.array: mask with shape (size, size, n_classes)
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if map_func is not None:
        mask = map_func(mask)

    mask = np.stack((mask,)*3, axis=-1)

    mask = common_ops(mask, size=size, CROP=CROP, FLIP_HORIZONTAL=False, FLIP_VERTICAL=False,
                      interpolation="nearest")[:, :, 0]

    mask[mask >= n_classes] = n_classes

    return mask


def load_img(path, size, CROP=None, BRIGHTNESS=False, FLIP_HORIZONTAL=False, 
             FLIP_VERTICAL=False):
    """ returns an image with shape (size, size, n_channels)

    Args:
        imgPath (str): path to the image
        size (tuple, optional): size of the image. Defaults to (256, 256).
        n_channels (int, optional): number of channels. Defaults to 3.
        SEED (tuple, optional): SEED for random crop. Defaults to None.
        crop (bool, optional): crop the image. Defaults to False.
        cropRandom (bool, optional): crop the image randomly. Defaults to False.

    Returns:
        np.array: image with shape (size, size, n_channels)
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return common_ops(img, size=size, CROP=CROP, FLIP_HORIZONTAL=FLIP_HORIZONTAL,
                      FLIP_VERTICAL=FLIP_VERTICAL, BRIGHTNESS=BRIGHTNESS,
                      interpolation="bilinear")
