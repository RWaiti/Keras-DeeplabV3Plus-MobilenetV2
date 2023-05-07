from os import environ, path
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

if False: # if True: turn off GPU
    environ["CUDA_VISIBLE_DEVICES"] = "-1"
    gpu = False
else:
    gpu = True

from IPython.display import clear_output
from tensorflow import keras
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

if gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_virtual_device_configuration(
        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 8))])
        print(gpus)
    except IndexError:
        print("gpu not appearing")

print(tf.__version__)

SEED = 42
BATCH_SIZE = 32
IMG_SIZE = (256, 256)

from glob import glob
from utils import data_utils

DATA_DIR = "utils/split/_split_/_type_"

train_X = sorted(glob(DATA_DIR.replace("_split_", "TrainVal").replace("_type_", "img") + "\\*"))
train_Y = sorted(glob(DATA_DIR.replace("_split_", "TrainVal").replace("_type_", "mask") + "\\*"))

train_X += sorted(glob(DATA_DIR.replace("_split_", "TrainVal").replace("_type_", "imgFog") + "\\*"))
train_Y += sorted(glob(DATA_DIR.replace("_split_", "TrainVal").replace("_type_", "mask") + "\\*"))

# Train image has 1/3 chance to be flipped
# Train image has 1/3 chance to be cropped
trainDataset = data_utils.load_dataset(
    train_X, train_Y, BATCH_SIZE=BATCH_SIZE, IMAGE_SIZE=IMG_SIZE, REMAP="binary", CROP=True, flip=True)

test_X = sorted(glob(DATA_DIR.replace("_split_", "Test").replace("_type_", "img") + "\\*"))
test_Y = sorted(glob(DATA_DIR.replace("_split_", "Test").replace("_type_", "mask") + "\\*"))
testData = data_utils.load_testset(
    test_X, test_Y, IMAGE_SIZE=IMG_SIZE, BATCH_SIZE=1, REMAP="binary")

print(len(train_X), len(train_Y))
print(len(test_X), len(test_Y))

from utils import lossesAccuracyfuncs
from utils import model_utils
import models

lr = 1e-4
alpha = 0.5
deeplayer = 12
l1 = l2 = 1e-5
n_classes = 2
epochs = 3000

mobileLayers = {"shallowLayer": "block_2_project_BN",
                "deepLayer": f"block_{deeplayer}_project_BN"}
    
fold = "final"
MODEL_NAME = "deeplabV3+_mobileNetV2"
SAVE_PATH = f"model-saved\\{MODEL_NAME}\\deeplayer_{deeplayer}_alpha_{alpha}_lr_{lr}_regularizer_l1_{l1}_l2_{l2}\\{fold}"

print(MODEL_NAME)
print(f"lr: {lr} | alpha: {alpha} | deeplayer: {deeplayer}")
print(f"fold: {fold} | l1: {l1} | l2: {l2} | classes: {n_classes}")

model = models.deeplabV3(
    imageSize=IMG_SIZE, nClasses=n_classes, alpha=alpha, mobileLayers=mobileLayers,
    kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2), SEED=SEED)

losses = lossesAccuracyfuncs.Losses_n_Metrics()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=n_classes),
    metrics=[losses.diceAccuracy, losses.jaccardDistance], sample_weight_mode="temporal")

history = model.fit(
    x=trainDataset, validation_data=testData, batch_size=BATCH_SIZE, verbose=2,
    epochs=epochs, callbacks=model_utils.callbacks_func(savePath=SAVE_PATH, monitor="val_loss"))
