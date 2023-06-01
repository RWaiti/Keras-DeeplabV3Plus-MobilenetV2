from os import environ, path
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

from sklearn.model_selection import ParameterSampler
from utils import lossesAccuracyfuncs
from utils import model_utils
from utils import data_utils
import tensorflow as tf
from numpy import ceil
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

SEED = 42
BATCH_SIZE = 2
IMG_SIZE = (240, 320)
DATA_DIR = "utils/split_3/_split_/_type_"
MODEL_NAME = "deeplabV3+_mobileNetV2"

train_val_X = sorted(glob(DATA_DIR.replace("_split_", "TrainVal").replace("_type_", "img") + "/*"))
train_val_Y = sorted(glob(DATA_DIR.replace("_split_", "TrainVal").replace("_type_", "mask") + "/*"))

test_X = sorted(glob(DATA_DIR.replace("_split_", "Test").replace("_type_", "img") + "/*"))
test_Y = sorted(glob(DATA_DIR.replace("_split_", "Test").replace("_type_", "mask") + "/*"))

print(len(train_val_X), len(train_val_Y))
print(len(test_X), len(test_Y))

l1 = 1e-6
l2 = 1e-4
lr = 1e-3
alpha = .5
epochs = 750
n_classes = 2
deeplayer = 12

mobileLayers = {"shallowLayer": "block_2_project_BN",
                "deepLayer": f"block_{deeplayer}_project_BN"}

losses = lossesAccuracyfuncs.Losses_n_Metrics()

train_val_folds = data_utils.load_datasetCV(
    train_val_X, train_val_Y, IMAGE_SIZE=IMG_SIZE, BATCH_SIZE=BATCH_SIZE,
    N_FOLDS=5, SEED=SEED)

for fold, (trainDataset, valDataset, n_classes) in enumerate(train_val_folds):
    SAVE_PATH = f"model-saved/{MODEL_NAME}/NEW_2/kfold/layer_{deeplayer}_alpha_{alpha}_regularizer_l1_{l1}_l2_{l2}/{fold}"

    print(MODEL_NAME)
    print(f"lr: {lr} | alpha: {alpha} | deeplayer: {deeplayer}")
    print(f"fold: {fold} | l1: {l1} | l2: {l2} | classes: {n_classes}")

    model = models.deeplabV3(
        imageSize=IMG_SIZE, nClasses=n_classes, alpha=alpha, mobileLayers=mobileLayers,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2), SEED=SEED)

    model.compile(
        optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=lr),
        loss=losses.loss_dice_coef,
        metrics=[losses.dice_coef, losses.iou_coef], sample_weight_mode="temporal")

    history = model.fit(
        x=trainDataset, validation_data=valDataset, batch_size=BATCH_SIZE, verbose=2,
        epochs=epochs, callbacks=model_utils.callbacks_func(savePath=SAVE_PATH, monitor="val_loss"), 
        workers=3, max_queue_size=30)

search_space = {
    "l2": [1e-2, 1e-3, 1e-4],
    "loss": [losses.loss_dice_coef, losses.loss_iou_coef]}

all_iter = 3 * 2
n_iter = int(ceil(all_iter * 1.)) # 100%

print(all_iter, n_iter)

search_space = list(ParameterSampler(search_space, n_iter=n_iter, random_state=SEED))

for _dict in search_space:
    print(_dict)

for _dict in search_space:
    l2 = _dict["l2"]
    loss = _dict["loss"]

    mobileLayers = {
        "shallowLayer": "block_2_project_BN",
        "deepLayer": f"block_{deeplayer}_project_BN"}

    # Train image has 1/3 chance to be flipped
    # Train image has 1/3 chance to be cropped
    # Train image has 1/3 chance to be fog
    trainDataset = data_utils.load_dataset(
        train_val_X, train_val_Y, BATCH_SIZE=BATCH_SIZE, IMAGE_SIZE=IMG_SIZE)

    testData = data_utils.load_testset(
        test_X, test_Y, IMAGE_SIZE=IMG_SIZE, BATCH_SIZE=BATCH_SIZE)

    SAVE_PATH = f"model-saved/{MODEL_NAME}/NEW_2/GRID_SEARCH/layer_{deeplayer}_{loss.__name__}_alpha_{alpha}_regularizer_l1_{l1}_l2_{l2}/grid_search"

    print(MODEL_NAME)
    print(f"lr: {lr} | alpha: {alpha} | deeplayer: {deeplayer}")
    print(f"l1: {l1} | l2: {l2} | classes: {n_classes}")

    model = models.deeplabV3(
        imageSize=IMG_SIZE, nClasses=n_classes, alpha=alpha, mobileLayers=mobileLayers,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2), SEED=SEED)

    model.compile(
        optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=lr),
        loss=loss,
        metrics=[losses.dice_coef, losses.iou_coef], sample_weight_mode="temporal")

    history = model.fit(
        x=trainDataset, validation_data=testData, batch_size=BATCH_SIZE, verbose=2,
        epochs=epochs, callbacks=model_utils.callbacks_func(savePath=SAVE_PATH, monitor="val_loss"), 
        workers=3, max_queue_size=30)