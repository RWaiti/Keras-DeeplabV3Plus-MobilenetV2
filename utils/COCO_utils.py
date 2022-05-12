from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pycocotools.coco import COCO
from random import shuffle
import tensorflow as tf
from PIL import Image
import numpy as np
import os


def COCO_data(DATASET_DIR="", TYPE="train", YEAR="2017", catList=None, BATCH_SIZE=4, IMAGE_SIZE=256, FROM=0, TO=-1):
    DATA_DIR = os.path.join(DATASET_DIR, TYPE + YEAR)
    ANNOTATION_DIR = os.path.join(DATASET_DIR, "annotations/instances_"+TYPE)

    coco = COCO(ANNOTATION_DIR + YEAR + ".json")
    catIdsList = []
    print(catList)
    if catList is not None and isinstance(catList, list):
        catIdsList = coco.getCatIds(supNms=catList)
        if len(catIdsList) == 0:
            catIdsList = coco.getCatIds(catNms=catList)
            print(catIdsList)
        else:
            print(catIdsList)
        NUM_CLASSES = len(catIdsList) + 1
    else:
        NUM_CLASSES = 91

    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)
    shuffle(images)
    images = images[FROM:TO]

    if catList is not None and isinstance(catList, list):
        newImages = []
        for img in images:
            flag = False
            annIds = coco.getAnnIds(imgIds=img["id"], catIds=catIds)
            anns = coco.loadAnns(annIds)
            for j in anns:
                if j["category_id"] in catIdsList:
                    flag = True
            if flag:
                newImages.append(img)
        images = newImages
        newImages = None

    dataset = COCO_datagenerator(
        images, coco, DATA_DIR, catIds, catIdsList=catIdsList, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

    return dataset, len(images), NUM_CLASSES


def COCO_datagenerator(dataset=None, coco=None, DATA_DIR=None, catIds=None, catIdsList=[], batch_size=4, image_size=256):
    pos = 0

    while True:
        image = np.zeros(
            (batch_size, image_size, image_size, 3), dtype=np.float32)
        mask = np.zeros(
            (batch_size, image_size, image_size, 1), dtype=np.float32)

        for i in range(pos, pos + batch_size):
            img = dataset[i]

            aux = tf.io.read_file(DATA_DIR + "/" + img["file_name"])
            aux = tf.image.decode_image(aux, channels=3)
            aux.set_shape([None, None, 3])
            image[i - pos] = tf.image.resize(images=aux,
                                             size=[image_size, image_size]) / 127.5 - 1

            annIds = coco.getAnnIds(imgIds=img["id"], catIds=catIds)
            anns = coco.loadAnns(annIds)

            maskAux = np.zeros(
                (img['height'], img['width'], 1), dtype=np.float32)

            for j in anns:
                if catIdsList is None:
                    maskAux[:, :, 0] = np.maximum(coco.annToMask(j) * j["category_id"],
                                                  maskAux[:, :, 0])
                elif j["category_id"] in catIdsList:
                    maskAux[:, :, 0] = np.maximum(coco.annToMask(j) * (catIdsList.index(j["category_id"]) + 1),
                                                  maskAux[:, :, 0])

            maskAux = tf.convert_to_tensor(maskAux)
            maskAux.set_shape([None, None, 1])
            mask[i - pos] = tf.image.resize(images=maskAux,
                                            size=[image_size, image_size])

        pos += batch_size

        if pos + batch_size >= len(dataset):
            pos = 0
            shuffle(dataset)

        yield image, mask


# https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-2-of-2-c0d1f593096a

def COCO_Aug(gen, augGeneratorArgs):
    augGeneratorArgs_mask = augGeneratorArgs.copy()
    _ = augGeneratorArgs_mask.pop('brightness_range', None)

    image_gen = ImageDataGenerator(**augGeneratorArgs)
    mask_gen = ImageDataGenerator(**augGeneratorArgs_mask)

    for img, mask in gen:
        seed = np.random.choice(range(9999))

        g_x = image_gen.flow(((img + 1) * 255).astype(np.int16).astype(np.float32) / 2,
                             batch_size=img.shape[0],
                             seed=seed,
                             shuffle=True)
        g_y = mask_gen.flow(mask,
                            batch_size=mask.shape[0],
                            seed=seed,
                            shuffle=True)

        yield next(g_x) / 127.5 - 1, next(g_y)
