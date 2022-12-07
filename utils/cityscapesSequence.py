import tensorflow as tf
import cv2
import numpy as np
import random
from sklearn.utils import class_weight
import utils.imageUtils as imageUtils

#          name                     id  trainId category        catId   catId_mode  binary_mode
labels = [['unlabeled',             0,  255,    'void',         0],   # 7           2
          ['ego vehicle',           1,  255,    'void',         0],   # 7           2
          ['rectification border',  2,  255,    'void',         0],   # 7           2
          ['out of roi',            3,  255,    'void',         0],   # 7           2
          ['static',                4,  255,    'void',         0],   # 7           2
          ['dynamic',               5,  255,    'void',         0],   # 7           2
          ['ground',                6,  255,    'void',         0],   # 7           2
          ['road',                  7,    0,    'ground',       1],   # 0           0
          ['sidewalk',              8,    1,    'ground',       1],   # 0           0
          ['parking',               9,  255,    'ground',       1],   # 0           0
          ['rail track',            10, 255,    'ground',       1],   # 0           0
          ['building',              11,   2,    'construction', 2],   # 1           1
          ['wall',                  12,   3,    'construction', 2],   # 1           1
          ['fence',                 13,   4,    'construction', 2],   # 1           1
          ['guard rail',            14, 255,    'construction', 2],   # 1           1
          ['bridge',                15, 255,    'construction', 2],   # 1           1
          ['tunnel',                16, 255,    'construction', 2],   # 1           1
          ['pole',                  17,   5,    'object',       3],   # 2           1
          ['polegroup',             18, 255,    'object',       3],   # 2           1
          ['traffic light',         19,   6,    'object',       3],   # 2           1
          ['traffic sign',          20,   7,    'object',       3],   # 2           1
          ['vegetation',            21,   8,    'nature',       4],   # 3           1
          ['terrain',               22,   9,    'nature',       4],   # 3           1
          ['sky',                   23,  10,    'sky',          5],   # 4           1
          ['person',                24,  11,    'human',        6],   # 5           1
          ['rider',                 25,  12,    'human',        6],   # 5           1
          ['car',                   26,  13,    'vehicle',      7],   # 6           1
          ['truck',                 27,  14,    'vehicle',      7],   # 6           1
          ['bus',                   28,  15,    'vehicle',      7],   # 6           1
          ['caravan',               29, 255,    'vehicle',      7],   # 6           1
          ['trailer',               30, 255,    'vehicle',      7],   # 6           1
          ['train',                 31,  16,    'vehicle',      7],   # 6           1
          ['motorcycle',            32,  17,    'vehicle',      7],   # 6           1
          ['bicycle',               33,  18,    'vehicle',      7],   # 6           1
          ['license plate',         -1,  -1,    'vehicle',      7]]   # 6           1


class CitySequence(tf.keras.utils.Sequence):
    def __init__(self, xDir=None, yDir=None, batchSize=4, imageSize=256, horizontalFlip=False,
                 verticalFlip=False, brightness=0, blur=0, contrast=0, crop=False, remap=""):
        self.xDir, self.yDir = xDir, yDir

        self.labels = labels

        self.batchSize = batchSize
        self.imageSize = imageSize

        self.horizontalFlip = horizontalFlip
        self.verticalFlip = verticalFlip
        self.brightness = brightness
        self.contrast = contrast
        self.blur = blur
        self.crop = crop

        if remap == "catId":
            self.nClasses = 7
            self.remapMethod = self.__catId__
        elif remap == "binary":
            self.nClasses = 2
            self.remapMethod = self.__binary__
        else:
            self.nClasses = 19
            self.remapMethod = self.__withoutRemap__

    def __len__(self):
        return np.ceil(len(self.xDir) / self.batchSize).astype(np.int64)

    def __augmentation__(self, img, mask):
        if self.blur > 0 and random.randint(0, 1):
            img = cv2.GaussianBlur(img, (self.blur, self.blur), 0)
        if self.horizontalFlip and random.randint(0, 2) == 2:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        if self.verticalFlip and random.randint(0, 2) == 2:
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)
        if self.brightness > 0 and random.randint(0, 2) == 2:
            img = tf.image.adjust_brightness(img, self.brightness).numpy()
            img[img > 1] = 1.
        if tf.is_tensor(mask):
            return img, mask.numpy()
        return img, mask

    def __withoutRemap__(self, mask):
        return mask

    def __catId__(self, mask):
        maskAux = np.zeros_like(mask, dtype=np.int8)

        for label in self.labels:
            maskAux[mask == label[1]] = label[4] - 1

        maskAux[maskAux == -1] = 7

        return maskAux

    def __binary__(self, mask):
        maskAux = np.zeros_like(mask, dtype=np.int8)

        for label in self.labels:
            if label[0] == 'ground' or label[3] == 'ground':
                maskAux[mask == label[1]] = 1
            elif label[4] == 0:
                maskAux[mask == label[1]] = 2
            else:
                maskAux[mask == label[1]] = 0

        return maskAux

    def __sampleWeights__(self, mask):
        maskAux = mask.flatten()
        auxClasses = np.unique(maskAux)

        if len(auxClasses):
            sampleWeight = np.ones(mask.shape)

            weightsVector = class_weight.compute_class_weight(class_weight="balanced",
                                                               classes=auxClasses,
                                                               y=maskAux)

            for j, weight in zip(auxClasses, weightsVector):
                sampleWeight[maskAux == int(j)] = weight

            sampleWeight[maskAux >= self.nClasses] = 0
        else:
            sampleWeight = np.zeros(maskAux.shape)

        return sampleWeight

    def __getitem__(self, idx):
        batchX = self.xDir[idx * self.batchSize:(idx + 1) * self.batchSize]
        batchY = self.yDir[idx * self.batchSize:(idx + 1) * self.batchSize]

        images = np.zeros(
            (self.batchSize, self.imageSize, self.imageSize, 3), dtype=np.float32)
        masks = np.zeros(
            (self.batchSize, self.imageSize * self.imageSize, 1), dtype=np.float32)
        sampleWeights = np.zeros(
            (self.batchSize, self.imageSize * self.imageSize, 1), dtype=np.float32)

        for i in range(len(batchX)):
            seed = (random.randint(1, 5), random.randint(1, 5))
            cropRandom = random.randint(0, 1)
            # Image
            img = imageUtils.getImgData(batchX[i], imgSize=(self.imageSize, self.imageSize), seed=seed, crop=self.crop, cropRandom=cropRandom)

            # Mask
            mask = imageUtils.getMaskData(batchY[i], imgSize=(self.imageSize, self.imageSize), nClasses=self.nClasses, mapFunc=self.remapMethod, seed=seed, crop=self.crop, cropRandom=cropRandom)

            # Augmentation
            img, mask = self.__augmentation__(img, np.expand_dims(mask, axis=-1))

            mask = mask.flatten()

            images[i] = img
            masks[i] = np.expand_dims(mask, axis=-1)
            sampleWeights[i] = np.expand_dims(self.__sampleWeights__(mask), axis=-1)

        sampleDicts = {"output": sampleWeights}
        return images, masks, sampleDicts

    # Shuffle data after each epoch if shuffle is True
    def on_epoch_end(self):
        toShuffle = list(zip(self.xDir, self.yDir))
        random.shuffle(toShuffle)
        self.xDir, self.yDir = zip(*toShuffle)
