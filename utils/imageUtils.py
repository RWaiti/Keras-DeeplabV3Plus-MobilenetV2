# Path: utils\imageUtils.py
import os
import cv2
import numpy as np
from glob import glob
from random import shuffle
from tensorflow import cast, io, float32
from tensorflow.image import stateless_random_crop, reshape, resize, decode_image


def getMaskData(maskPath, imgSize=(256, 256), nClasses=19, mapFunc=None, seed=None, crop=False, cropRandom=False):
    """ returns a mask with shape (imgSize, imgSize, nClasses)

    Args:
        maskPath (str): path to the mask
        imgSize (tuple, optional): size of the image. Defaults to (256, 256).
        nClasses (int, optional): number of classes. Defaults to 19.
        mapFunc (function, optional): function to map the mask. Defaults to None.

    Returns:
        np.array: mask with shape (imgSize, imgSize, nClasses)
    """
    mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)

    if crop and cropRandom:
        mask = stateless_random_crop(mask, (int(mask.shape[0] // 1.2),
                                            int(mask.shape[1] // 1.2)),
                                            seed=seed).numpy()
    mask = cv2.resize(mask, imgSize,
                        interpolation=cv2.INTER_NEAREST)

    if mapFunc is not None:
        mask = mapFunc(mask)
    else:
        mask[mask >= nClasses] = nClasses

    return mask

def getImgData(imgPath, imgSize=(256, 256), nChannels=3, seed=None, crop=False, cropRandom=False):
    """ returns an image with shape (imgSize, imgSize, nChannels)

    Args:
        imgPath (str): path to the image
        imgSize (tuple, optional): size of the image. Defaults to (256, 256).
        nChannels (int, optional): number of channels. Defaults to 3.
        seed (tuple, optional): seed for random crop. Defaults to None.
        crop (bool, optional): crop the image. Defaults to False.
        cropRandom (bool, optional): crop the image randomly. Defaults to False.

    Returns:
        np.array: image with shape (imgSize, imgSize, nChannels)
    """

    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if crop and cropRandom:
        img = stateless_random_crop(img, (int(img.shape[0] // 1.2),
                                          int(img.shape[1] // 1.2), 3),
                                          seed=seed).numpy()
    img = cv2.resize(img, (256, 256),
                        interpolation=cv2.INTER_AREA) / 127.5 - 1
    return img

def representativeDatasetGen(path=""):
    imagePath = os.path.join(path, "leftImg8bit", "train", "*", "*", "*_leftImg8bit.png")

    images = glob(imagePath)
    shuffle(images)
    images = images[0:75]

    for img in images:
        aux = io.read_file(img)
        aux = decode_image(aux, channels=3)
        aux.set_shape([None, None, 3])
        aux = resize(images=aux,
                              size=[256, 256]) / 127.5 - 1
        aux = cast(aux, float32)
        
        yield [reshape(aux, (1, aux.shape[0], aux.shape[1], aux.shape[2]))]