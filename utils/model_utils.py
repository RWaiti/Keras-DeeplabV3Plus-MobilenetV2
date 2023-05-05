from os import environ, path
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import callbacks

def callbacks_func(savePath, monitor="val_loss"):
    callbacksList = []

    callbacksList += [callbacks.EarlyStopping(
        monitor=monitor, min_delta=0.0001, patience=140)]

    callbacksList += [callbacks.ModelCheckpoint(
        savePath+".ckpt", monitor=monitor, verbose=1,
        save_best_only=True, mode="auto")]
    
    callbacksList += [callbacks.CSVLogger(filename=savePath+".csv")]

    callbacksList += [callbacks.ReduceLROnPlateau(
        monitor=monitor, factor=0.50, patience=45, verbose=0)]

    return callbacksList

from tensorflow import lite as tflite

def toTFlite(
        model, savePath, representativeDatasetGen,
        supportedOps = [tflite.OpsSet.SELECT_TF_OPS,
                        tflite.OpsSet.TFLITE_BUILTINS,
                        tflite.OpsSet.TFLITE_BUILTINS_INT8]):

    converter = tflite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tflite.Optimize.DEFAULT]
    converter.representative_dataset = representativeDatasetGen
    converter.target_spec.supported_ops = supportedOps
    tflite_model = converter.convert()

    with open(savePath + ".tflite", "wb") as f:
        f.write(tflite_model)

from glob import glob
from numpy import asarray
from random import shuffle
from tensorflow import cast, float32, reshape, Tensor
from cv2 import imread, cvtColor, IMREAD_COLOR, COLOR_BGR2RGB

class BatchGenerator():
    def __init__(self, images=None):
        assert isinstance(images, list)

        shuffle(images)
        self.images = images[0:75]

    def __call__(self):
        for img in self.images:
            img = imread(img, IMREAD_COLOR)
            img = cvtColor(img, COLOR_BGR2RGB)
            img = asarray(img, dtype="float32" ) / 127.5 - 1.

            yield [reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))]