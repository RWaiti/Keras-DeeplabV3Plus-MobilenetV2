import random
from os import mkdir
from glob import glob
from os.path import join, exists

from sklearn.model_selection import KFold

from utils.cityscapesSequence import CitySequence


def load_datasetCV(X, Y, BATCH_SIZE=1, IMAGE_SIZE=(256, 256), REMAP="binary", N_FOLDS=5, SEED=42, use_fog=False, flip=False):
    train_val_X, train_val_Y = data_shuffle(X, Y)
    kfolds = split_kfold(train_val_X, train_val_Y, nSplits=N_FOLDS, seed=SEED)

    for (train_X, train_Y, val_X, val_Y) in kfolds:
        train_dataset = CitySequence(
            train_X, train_Y, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
            remap=REMAP, use_fog=use_fog, flip=flip)

        val_dataset   = CitySequence(
            val_X, val_Y, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
            remap=REMAP)

        yield train_dataset, val_dataset, train_dataset.n_classes

def load_dataset(X, Y, BATCH_SIZE=1, IMAGE_SIZE=(256, 256), REMAP="binary", CROP=False, flip=False):
    return CitySequence(
        X, Y, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
            remap=REMAP, CROP=CROP, flip=flip)

def load_testset(X, Y, IMAGE_SIZE=(256, 256), BATCH_SIZE=1, REMAP="binary"):
    return CitySequence(X, Y, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, remap=REMAP)


def data_shuffle(x, y, seed=0) -> tuple[list, list]:
    """ Shuffle the data

    Args:
        x (list): list of paths to the data
        y (list): list of paths to the data ground truth
        seed (int, optional): seed for the random generator. Defaults to 0.

    Returns:
        list: list of shuffled data in the form (xData, yData)
    """
    random.seed(seed)
    to_shuffle = list(zip(x, y))
    random.shuffle(to_shuffle)
    return zip(*to_shuffle)

def split_kfold(X, Y, nSplits=5, seed=0):
    """ Split the data in train and validation folds

    Args:
        xData (list): list of paths to the data
        yData (list): list of paths to the data ground truth
        nSplits (int, optional): number of folds. Defaults to 5.
        seed (int, optional): seed for the random generator. Defaults to 0.

    Returns:
        list: list of train and validation folds in the form (xTrain, yTrain, xVal, yVal)
    """

    kf = KFold(n_splits=nSplits, shuffle=True, random_state=seed)

    for trainIdx, valIdx in kf.split(X):
        xTrain = [X[i] for i in trainIdx]
        yTrain = [Y[i] for i in trainIdx]
        xVal = [X[i] for i in valIdx]
        yVal = [Y[i] for i in valIdx]
        yield xTrain, yTrain, xVal, yVal

def create_folder(pathList):
    """ Generate all files in the list
        WARNING: Not tested to absolute paths "C:\\..." or paths above the notebook path "..\\..\\"

    Args:
        pathList (list): list containing paths to desired files.
        Ex. ["split\\train\\img\\"] == ["split, "split\\train, "split\\train\\img\\"]
            ["split\\train", "split\\val"] folder with two sub folders
    """
    for path in pathList:
        split = path.replace("/", "\\").split("\\")
        separator = "\\"
        files = [separator.join(split[:i+1]) for i in range(len(split))]

        for file in files:
            if exists(file):
                continue
            mkdir(file)
