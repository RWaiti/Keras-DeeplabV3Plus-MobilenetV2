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

from sklearn.model_selection import ParameterSampler

search_space = {
    "l1": [1e-4, 1e-5],
    "l2": [1e-5, 1e-6],
    "lr": [1e-3, 1e-4],
    "alpha": [.5, 1.],
    "deeplayer": [12],
    "epochs": [1000]}

all_iter = 2 * 2 * 2 * 2 * 1 * 1 * 1
n_iter = int(all_iter * 0.75)

print(all_iter, n_iter)

search_space = list(ParameterSampler(search_space, n_iter=n_iter, random_state=SEED))

for _dict in search_space:
    print(_dict)


from glob import glob
from utils import data_utils

DATA_DIR = "utils/split/_split_/_type_"

train_val_X = sorted(glob(DATA_DIR.replace("_split_", "TrainVal").replace("_type_", "img") + "\\*"))
train_val_Y = sorted(glob(DATA_DIR.replace("_split_", "TrainVal").replace("_type_", "mask") + "\\*"))

test_X = sorted(glob(DATA_DIR.replace("_split_", "Test").replace("_type_", "img") + "\\*"))
test_Y = sorted(glob(DATA_DIR.replace("_split_", "Test").replace("_type_", "mask") + "\\*"))
# test_data = data_utils.load_testset(test_X, test_Y, IMAGE_SIZE=IMG_SIZE, BATCH_SIZE=1, REMAP="binary")


print(len(train_val_X), len(train_val_Y))
print(len(test_X), len(test_Y))

from utils import lossesAccuracyfuncs
from utils import model_utils
import models

for _dict in search_space:
    lr = _dict["lr"]
    alpha = _dict["alpha"]
    deeplayer = _dict["deeplayer"]
    l1, l2 = _dict["l1"], _dict["l2"]

    mobileLayers = {"shallowLayer": "block_2_project_BN",
                    "deepLayer": f"block_{deeplayer}_project_BN"}
    
    # 20% of a train image be a fog image
    # 33% of a train image to be flipped
    train_val_folds = data_utils.load_dataset(
        train_val_X, train_val_Y, IMAGE_SIZE=IMG_SIZE, BATCH_SIZE=BATCH_SIZE, REMAP="binary",
        N_FOLDS=3, SEED=SEED, use_fog=True, flip=True)
    
    for fold, (trainDataset, valDataset, n_classes) in enumerate(train_val_folds):
        MODEL_NAME = "deeplabV3+_mobileNetV2"
        SAVE_PATH = f"model-saved\\{MODEL_NAME}\\deeplayer_{deeplayer}_alpha_{alpha}_lr_{lr}_regularizer_l1_{l1}_l2_{l2}\\fold_{fold}"

        print(MODEL_NAME)
        print(f"lr: {lr} | alpha: {alpha} | deeplayer: {deeplayer}")
        print(f"fold: {fold} | l1: {l1} | l2: {l2} | classes: {n_classes}")
        
        if path.exists(SAVE_PATH):
            continue

        model = models.deeplabV3(
            imageSize=IMG_SIZE, nClasses=n_classes, alpha=alpha, mobileLayers=mobileLayers,
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2), SEED=SEED)

        losses = lossesAccuracyfuncs.Losses_n_Metrics()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=n_classes),
            metrics=[losses.diceAccuracy, losses.jaccardDistance], sample_weight_mode="temporal")

        history = model.fit(
            x=trainDataset, validation_data=valDataset, batch_size=BATCH_SIZE, verbose=2,
            epochs=_dict["epochs"], callbacks=model_utils.callbacks_func(savePath=SAVE_PATH, monitor="val_loss"))

# ##### TFLITE Converter
# 


# representativeData = representativeDatasetGen(path="../cityscapes/alldata")
# supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]


