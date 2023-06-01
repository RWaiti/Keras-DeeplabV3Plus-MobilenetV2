from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import numpy as np
import tensorflow as tf
from copy import deepcopy
from utils import image_utils
from sklearn.utils import class_weight

tf.get_logger().setLevel("ERROR")

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

def binary_func(mask):
    aux_mask = np.zeros_like(mask, dtype=np.int8)

    for label in labels:
        if label[0] == 'ground' or label[3] == 'ground':
            aux_mask[mask == label[1]] = 1
        elif label[1] in [1, 2, 3]:
            aux_mask[mask == label[1]] = 2
        else:
            aux_mask[mask == label[1]] = 0

    return aux_mask

def binary_2_func(mask):
    aux_mask = np.zeros_like(mask, dtype=np.int8)

    for label in labels:
        if label[0] == 'ground' or label[3] == 'ground':
            aux_mask[mask == label[1]] = 1
        elif label[1] in [2, 3]:
            aux_mask[mask == label[1]] = 2
        else:
            aux_mask[mask == label[1]] = 0

    return aux_mask

def cat_id_func(mask):
    aux_mask = np.zeros_like(mask, dtype=np.int8)

    for label in labels:
        aux_mask[mask == label[1]] = label[4] - 1

    aux_mask[aux_mask == -1] = np.max(aux_mask) + 1

    return aux_mask

class CitySequence(tf.keras.utils.Sequence):
    def __init__(
            self, x_dir: list[str], y_dir: list[str], batch_size=4, image_size=(256, 256),
            remap="", BRIGHTNESS=False, CROP=False, VERTICAL_FLIP=False,
            HORIZONTAL_FLIP=False):
        self.x_dir, self.y_dir = x_dir, y_dir

        self.labels = labels
        self.BRIGHTNESS = BRIGHTNESS
        self.VERTICAL_FLIP = VERTICAL_FLIP
        self.HORIZONTAL_FLIP = HORIZONTAL_FLIP
        self.CROP = CROP

        self.batch_size = batch_size
        self.image_size = image_size

        if remap == "catId":
            self.n_classes = 7
            self.__remap_method = cat_id_func
        elif remap == "binary":
            self.n_classes = 2
            self.__remap_method = binary_func
        elif remap == "binary_2":
            self.n_classes = 2
            self.__remap_method = binary_2_func
        else:
            self.n_classes = 19
            self.__remap_method = lambda x: x

    def __len__(self) -> int:
        return np.ceil(len(self.x_dir) / self.batch_size).astype(np.int32) 

    def __sample_weights(self, mask: np.ndarray):
        uniques, counts = np.unique(mask, return_counts=True)
        samples_weights = np.zeros_like(mask, dtype=np.float32)

        counts[uniques >= self.n_classes] = 0.

        total = np.sum(counts)

        percent = counts / total

        for i, weight in zip(uniques, percent):
            samples_weights[mask == i] = weight

        return np.expand_dims(samples_weights, axis=-1)

    def __getitem__(self, idx):
        batchX = self.x_dir[idx * self.batch_size:(idx + 1) * self.batch_size]
        batchY = self.y_dir[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], 3), dtype=np.float32)
        masks = np.zeros((self.batch_size, self.image_size[0] * self.image_size[1], 1), dtype=np.float32)
        sample_weights = np.zeros((self.batch_size, self.image_size[0] * self.image_size[1], 1), dtype=np.float32)

        # expected already resized data
        for i, (img, mask) in enumerate(zip(batchX, batchY)):
            FLIP_HORIZONTAL = True if (self.HORIZONTAL_FLIP and random.random() < 0.20) else False
            FLIP_VERTICAL = True if (self.VERTICAL_FLIP and random.random() < 0.20) else False
            BRIGHTNESS = True if (self.BRIGHTNESS and random.random() < 0.20) else False
            CROP = None

            if (self.CROP and random.random() < 0.20):
                CROP_PERCENT = .75

                new_height, new_width = int(self.image_size[0] * CROP_PERCENT), int(self.image_size[1] * CROP_PERCENT)

                row_max = self.image_size[0]
                col_max = self.image_size[1]
                
                row_left = random.randint(0, row_max-new_height)
                row_right= new_height+row_left

                col_upper = random.randint(0, col_max-new_width)
                col_down = new_width+col_upper
                CROP = [[row_left, row_right], [col_upper, col_down]]

            images[i] = image_utils.load_img(
                img, self.image_size, CROP=CROP, FLIP_HORIZONTAL=FLIP_HORIZONTAL,
                FLIP_VERTICAL=FLIP_VERTICAL, BRIGHTNESS=BRIGHTNESS)

            mask = image_utils.load_mask(
                mask, self.image_size, n_classes=self.n_classes, CROP=CROP, 
                FLIP_HORIZONTAL=FLIP_HORIZONTAL, FLIP_VERTICAL=FLIP_VERTICAL).flatten()

            masks[i] = np.expand_dims(mask, axis=-1)
            sample_weights[i] = self.__sample_weights(mask)

        return images, masks, {"output": sample_weights}

    # Shuffle data after each epoch if shuffle is True
    def on_epoch_end(self):
        toShuffle = list(zip(self.x_dir, self.y_dir))
        random.shuffle(toShuffle)
        self.x_dir, self.y_dir = zip(*toShuffle)
