# Path: utils\image_utils.py
import os
import cv2
import numpy as np
from glob import glob
from random import shuffle
from tensorflow import cast, float32, reshape, Tensor
from tensorflow.image import stateless_random_crop, flip_left_right, resize

CROP_PERCENT = .85


def common_ops(img, size=None, SEED=None, CROP=False, FLIP=False, constant_values=0, interpolation="bilinear"):
    width, height, n_channels = img.shape

    new_width, new_height = int(width * CROP_PERCENT), int(height * CROP_PERCENT)
    pad_width, pad_height = (width - new_width) // 2, (height - new_height) // 2

    if CROP:
        img = stateless_random_crop(img, (new_width, new_height, n_channels), seed=SEED)
        # img = np.pad(img, [(pad_width, ), (pad_height, ), (0, )], mode='constant', constant_values=constant_values)
    if FLIP:
        img = flip_left_right(img)
    if size is not None:
        img = resize(img, size, interpolation)

    if isinstance(img, Tensor):
        img = img.numpy()

    return img.astype(np.uint8)


def load_mask(path, size=None, n_classes=19, map_func=None, SEED=None, CROP=False, FLIP=False):
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

    mask = np.stack((mask,)*3, axis=-1)

    mask = common_ops(mask, size=size, SEED=SEED, CROP=CROP, FLIP=FLIP,
                      constant_values=n_classes, interpolation="nearest")[:, :, 0]

    if map_func is not None:
        mask = map_func(mask)

    mask[mask >= n_classes] = n_classes

    return mask


def load_img(path, size=None, SEED=None, CROP=False, FLIP=False):
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

    return common_ops(img, size=size, SEED=SEED, CROP=CROP, FLIP=FLIP,
                      constant_values=0, interpolation="bilinear")


def representative_datagen(path="", size=[256, 256]):
    paths = os.path.join(path, "leftImg8bit", "train", "*", "*", "*_leftImg8bit.png")

    images = glob(paths)
    shuffle(images)
    images = images[0:75]

    for img in images:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize(images=img, size=size, interpolation="bilinear") / 127.5 - 1.
        img = cast(img, float32)

        yield [reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))]
