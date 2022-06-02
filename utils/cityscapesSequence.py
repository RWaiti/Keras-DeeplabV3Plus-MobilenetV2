import tensorflow as tf
import cv2
import numpy as np
import random
from sklearn.utils import class_weight

#          name                     id  trainId category        catId
labels = [['unlabeled',             0,  255,    'void',         0],   # -1 -> 7
          ['ego vehicle',           1,  255,    'void',         0],   # -1 -> 7
          ['rectification border',  2,  255,    'void',         0],   # -1 -> 7
          ['out of roi',            3,  255,    'void',         0],   # -1 -> 7
          ['static',                4,  255,    'void',         0],   # -1 -> 7
          ['dynamic',               5,  255,    'void',         0],   # -1 -> 7
          ['ground',                6,  255,    'void',         0],   # -1 -> 7
          ['road',                  7,    0,    'ground',       1],   # 0
          ['sidewalk',              8,    1,    'ground',       1],   # 0
          ['parking',               9,  255,    'ground',       1],   # 0
          ['rail track',            10, 255,    'ground',       1],   # 0
          ['building',              11,   2,    'construction', 2],   # 1
          ['wall',                  12,   3,    'construction', 2],   # 1
          ['fence',                 13,   4,    'construction', 2],   # 1
          ['guard rail',            14, 255,    'construction', 2],   # 1
          ['bridge',                15, 255,    'construction', 2],   # 1
          ['tunnel',                16, 255,    'construction', 2],   # 1
          ['pole',                  17,   5,    'object',       3],   # 2
          ['polegroup',             18, 255,    'object',       3],   # 2
          ['traffic light',         19,   6,    'object',       3],   # 2
          ['traffic sign',          20,   7,    'object',       3],   # 2
          ['vegetation',            21,   8,    'nature',       4],   # 3
          ['terrain',               22,   9,    'nature',       4],   # 3
          ['sky',                   23,  10,    'sky',          5],   # 4
          ['person',                24,  11,    'human',        6],   # 5
          ['rider',                 25,  12,    'human',        6],   # 5
          ['car',                   26,  13,    'vehicle',      7],   # 6
          ['truck',                 27,  14,    'vehicle',      7],   # 6
          ['bus',                   28,  15,    'vehicle',      7],   # 6
          ['caravan',               29, 255,    'vehicle',      7],   # 6
          ['trailer',               30, 255,    'vehicle',      7],   # 6
          ['train',                 31,  16,    'vehicle',      7],   # 6
          ['motorcycle',            32,  17,    'vehicle',      7],   # 6
          ['bicycle',               33,  18,    'vehicle',      7],   # 6
          ['license plate',         -1,  -1,    'vehicle',      7],   # 6
          ]


class CitySequence(tf.keras.utils.Sequence):
    def __init__(self, x_dir=None, y_dir=None, batch_size=4, image_size=256, horizontal_flip=False,
                 vertical_flip=False, brightness=0, blur=0, contrast=0, crop=False, catId=False,
                 binary=False):
        self.x, self.y = x_dir, y_dir
        self.x_y_len = [len(x_dir), len(y_dir)]

        self.batch_size = batch_size
        self.image_size = image_size
        self.catId = catId

        if catId and binary:
            raise Exception("remap and binary cannot be used at the same time")

        if catId:
            self.num_classes = 7
            self.remap_func = self._catId
        elif binary:
            self.num_classes = 2
            self.remap_func = self._binary
        else:
            self.num_classes = 19
            self.remap_func = self._without_remap

        self.labels = labels

        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness = brightness
        self.contrast = contrast
        self.blur = blur
        self.crop = crop

    def __len__(self):
        return np.ceil(len(self.x) / self.batch_size).astype(np.int64)

    def augmentation(self, img, mask):
        if self.blur > 0 and random.randint(0, 1):
            img = cv2.GaussianBlur(img, (self.blur, self.blur), 0)
        if self.horizontal_flip and random.randint(0, 2) == 2:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        if self.vertical_flip and random.randint(0, 2) == 2:
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)
        if self.brightness > 0 and random.randint(0, 2) == 2:
            img = tf.image.adjust_brightness(img, self.brightness).numpy()
            img[img > 1] = 1.
        if tf.is_tensor(mask):
            return img, mask.numpy()
        return img, mask

    def _without_remap(self, mask):
        mask[mask >= self.num_classes] = self.num_classes

        return mask

    def _catId(self, mask):

        maskAux = np.zeros_like(mask, dtype=np.int8)

        for label in self.labels:
            if label[4] == 0 and label[0] != 'ground':
                maskAux[mask == label[1]] = self.num_classes
            else:
                maskAux[mask == label[1]] = label[4] - 1

        return maskAux

    def _binary(self, mask):
        maskAux = np.zeros_like(mask, dtype=np.int8)

        for label in self.labels:
            if label[4] == 1 or label[0] == 'ground':
                maskAux[mask == label[1]] = 1
            else:
                maskAux[mask == label[1]] = 0

        return maskAux

    def _sample_weight(self, mask):
        maskAux = mask.flatten()
        aux_classes = np.unique(maskAux)

        if len(aux_classes):
            sample_weight = np.ones(mask.shape)

            weights_vector = class_weight.compute_class_weight(class_weight="balanced",
                                                               classes=aux_classes,
                                                               y=maskAux)

            for j, weight in zip(aux_classes, weights_vector):
                sample_weight[maskAux == int(j)] = weight

            sample_weight[maskAux >= self.num_classes] = 0
        else:
            sample_weight = np.zeros(maskAux.shape)

        return sample_weight

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        masks = np.zeros(
            (self.batch_size, self.image_size * self.image_size, 1), dtype=np.float32)
        sample_weights = np.zeros(
            (self.batch_size, self.image_size * self.image_size, 1), dtype=np.float32)

        for i in range(len(batch_x)):
            seed = (random.randint(1, 5), random.randint(1, 5))
            crop = random.randint(0, 1)
            # Image
            img = cv2.imread(batch_x[i], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.crop and crop:
                img = tf.image.stateless_random_crop(img, (int(img.shape[0] // 1.2),
                                                           int(img.shape[1] // 1.2), 3),
                                                     seed=seed).numpy()
            img = cv2.resize(img, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_AREA) / 127.5 - 1

            # Mask
            mask = cv2.imread(batch_y[i], cv2.IMREAD_GRAYSCALE)
            if self.crop and crop:
                mask = tf.image.stateless_random_crop(mask, (int(mask.shape[0] // 1.2),
                                                             int(mask.shape[1] // 1.2)),
                                                      seed=seed).numpy()
            mask = cv2.resize(mask, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_NEAREST)

            mask = self.remap_func(mask)

            # Augmentation
            img, mask = self.augmentation(img, np.expand_dims(mask, axis=-1))

            mask = mask.flatten()

            images[i] = img
            masks[i] = np.expand_dims(mask, axis=-1)
            sample_weights[i] = np.expand_dims(self._sample_weight(mask), axis=-1)

        sample_dicts = {"output": sample_weights}
        return images, masks, sample_dicts

    def on_epoch_end(self):
        to_shuffle = list(zip(self.x, self.y))
        random.shuffle(to_shuffle)
        self.x, self.y = zip(*to_shuffle)
