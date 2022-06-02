import tensorflow as tf
import cv2
import numpy as np
import random
from sklearn.utils import class_weight


class PASCALSequence(tf.keras.utils.Sequence):
    def __init__(self, x_dir=None, y_dir=None, batch_size=4, image_size=256, horizontal_flip=False,
                 vertical_flip=False, brightness=0, blur=0, contrast=0, crop=False):
        self.x, self.y = x_dir, y_dir
        self.x_y_len = [len(x_dir), len(y_dir)]

        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = 21

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
            img[img > 1] = 1
        if tf.is_tensor(mask):
            return img, mask.numpy()
        return img, mask

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        masks = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 1), dtype=np.int32)
        sample_weights = np.zeros(
            (self.batch_size, self.image_size, self.image_size), dtype=np.float32)

        for i in range(len(batch_x)):
            # Image
            img = cv2.imread(batch_x[i], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_AREA) / 127.5 - 1

            # Mask
            mask = cv2.imread(batch_y[i], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_NEAREST).astype(np.int32)
            mask[mask >= self.num_classes] = 0

            # Augmentation
            img, mask = self.augmentation(img, np.expand_dims(mask, axis=-1))

            # Sample Weights
            maskAux = mask.flatten()
            aux_classes = np.unique(maskAux)

            if len(aux_classes):
                sample_weight = np.ones(mask.shape)

                weights_vector = class_weight.compute_class_weight(class_weight="balanced",
                                                                   classes=aux_classes,
                                                                   y=maskAux)

                for j, weight in zip(aux_classes, weights_vector):
                    sample_weight[mask == int(j)] = weight
                sample_weight[mask >= self.num_classes] = 0
            else:
                sample_weight = np.zeros(mask.shape)

            images[i, :, :, :] = img
            masks[i, :, :, :] = mask
            sample_weights[i, :, :] = sample_weight[:, :, 0]

        return images, masks, sample_weights

    def on_epoch_end(self):
        to_shuffle = list(zip(self.x, self.y))
        random.shuffle(to_shuffle)
        self.x, self.y = zip(*to_shuffle)
