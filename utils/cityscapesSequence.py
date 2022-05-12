import tensorflow as tf
import cv2
import numpy as np
import random
from sklearn.utils import class_weight


class CitySequence(tf.keras.utils.Sequence):
    def __init__(self, x_dir=None, y_dir=None, batch_size=4, image_size=256, horizontal_flip=False,
                 vertical_flip=False, brightness=0, blur=0, contrast=0, crop=False):
        self.x, self.y = x_dir, y_dir
        self.len = [len(x_dir), len(y_dir)]
        self.on_epoch_end()

        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = 19

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
        if self.horizontal_flip and random.randint(0, 1):
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        if self.vertical_flip and random.randint(0, 1):
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        if self.brightness > 0 and random.randint(0, 1):
            if random.randint(0, 1):
                img = tf.image.adjust_brightness(img, self.brightness)
            else:
                img = tf.image.adjust_brightness(img, -self.brightness)
        if self.contrast > 0 and random.randint(0, 1):
            if random.randint(0, 1):
                img = tf.image.adjust_contrast(img, self.contrast)
            else:
                img = tf.image.adjust_contrast(img, -self.contrast)
        if tf.is_tensor(mask):
            return img, mask.numpy().astype(np.int32)
        return img, mask.astype(np.int32)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        masks = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 1), dtype=np.int32)
        sample_weights = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 1), dtype=np.float32)

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
            img = cv2.resize(img, (self.image_size, self.image_size)).astype(np.float32) / 127.5 - 1

            # Mask
            mask = cv2.imread(batch_y[i], cv2.IMREAD_GRAYSCALE)
            if self.crop and crop:
                mask = tf.image.stateless_random_crop(mask, (int(mask.shape[0] // 1.2),
                                                             int(mask.shape[1] // 1.2)),
                                                      seed=seed).numpy()
            mask = cv2.resize(mask, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_NEAREST)

            mask[mask >= self.num_classes] = 0

            # Augmentation
            images[i], masks[i] = self.augmentation(img, np.expand_dims(mask, -1))

            # Sample Weights
            mask_aux = masks[i].flatten()
            aux_classes = np.unique(mask_aux)

            if len(aux_classes):
                sample_weight = np.ones(masks[i].shape)

                weights_vector = class_weight.compute_class_weight(class_weight="balanced",
                                                                   classes=aux_classes,
                                                                   y=mask_aux)

                for j, weight in zip(aux_classes, weights_vector):
                    sample_weight[masks[i] == int(j)] = weight
            else:
                sample_weight = np.zeros(masks[i].shape)

            sample_weights[i] = sample_weight

        sample_dict = {"output_mask": sample_weights}
        return images, masks, sample_dict

    def on_epoch_end(self):
        to_shuffle = list(zip(self.x, self.y))
        random.shuffle(to_shuffle)
        self.x, self.y = zip(*to_shuffle)
