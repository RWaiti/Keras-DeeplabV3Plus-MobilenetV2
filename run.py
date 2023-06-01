from os import environ, path
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

from utils import lossesAccuracyfuncs
from utils import model_utils
from utils import data_utils
import tensorflow as tf
from glob import glob
import models

tf.get_logger().setLevel("ERROR")

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 6))])
    print(gpus[0])
except IndexError:
    print("gpu not appearing")

DATA_DIR = "utils/split_3/_split_/_type_"

train_X = sorted(glob(DATA_DIR.replace("_split_", "TrainVal").replace("_type_", "img") + "/*"))  + \
    sorted(glob(DATA_DIR.replace("_split_", "TrainVal").replace("_type_", "imgFog0.01") + "/*")) + \
    sorted(glob(DATA_DIR.replace("_split_", "TrainVal").replace("_type_", "imgFog0.02") + "/*"))
train_Y = sorted(glob(DATA_DIR.replace("_split_", "TrainVal").replace("_type_", "mask") + "/*")) * 3

test_X = sorted(glob(DATA_DIR.replace("_split_", "Test").replace("_type_", "img") + "/*"))
test_Y = sorted(glob(DATA_DIR.replace("_split_", "Test").replace("_type_", "mask") + "/*"))

print(len(train_X), len(train_Y))
print(len(test_X), len(test_Y))

def runModel(
        IMG_SIZE, _deeplayer, _loss, _lr, _l1, _l2, _alpha, _n_classes, batchSize,
        VERTICAL_FLIP):

    MODEL_NAME = "deeplabV3+_mobileNetV2"

    epochs = 1000
    SEED = 42

    mobileLayers = {"shallowLayer": "block_2_project_BN",
                "deepLayer": f"block_{_deeplayer}_project_BN"}

    SAVE_PATH = f"model-saved/{MODEL_NAME}/NEW_2/LAST_MODEL/VERTICAL_FLIP_{VERTICAL_FLIP}/last_model"

    # Train image has 20% chance to be vertically flipped
    # Train image has 20% chance to be horizontally flipped
    # Train image has 20% chance to be cropped

    trainDataset = data_utils.load_dataset(
        train_X, train_Y, BATCH_SIZE=batchSize, IMAGE_SIZE=IMG_SIZE, CROP=True,
        HORIZONTAL_FLIP=True, VERTICAL_FLIP=VERTICAL_FLIP, BRIGHTNESS=True)

    testData = data_utils.load_testset(
        test_X, test_Y, IMAGE_SIZE=IMG_SIZE, BATCH_SIZE=batchSize)

    model = models.deeplabV3(
        imageSize=IMG_SIZE, nClasses=_n_classes, alpha=_alpha, mobileLayers=mobileLayers,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=_l1, l2=_l2), SEED=SEED)

    losses = lossesAccuracyfuncs.Losses_n_Metrics()

    model.compile(
        optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=_lr),
        loss=_loss,
        metrics=[losses.dice_coef, losses.iou_coef], sample_weight_mode="temporal")

    history = model.fit(
        x=trainDataset, validation_data=testData, batch_size=batchSize, verbose=2,
        epochs=epochs, callbacks=model_utils.callbacks_func(savePath=SAVE_PATH, monitor="val_loss"), 
        workers=3, max_queue_size=30)

l1 = 1e-6
l2 = 1e-3
lr = 1e-3
alpha = .5
n_classes = 2
batchSize = 2
deeplayer = 12
IMG_SIZE = (240, 320)
loss = lossesAccuracyfuncs.Losses_n_Metrics().loss_iou_coef

VERTICAL_FLIP = True
runModel(IMG_SIZE, deeplayer, loss, lr, l1, l2, alpha, n_classes, batchSize, VERTICAL_FLIP)

VERTICAL_FLIP = False
runModel(IMG_SIZE, deeplayer, loss, lr, l1, l2, alpha, n_classes, batchSize, VERTICAL_FLIP)