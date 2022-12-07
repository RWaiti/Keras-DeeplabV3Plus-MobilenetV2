from sklearn.model_selection import KFold
from os.path import join
from glob import glob
import random

def getXY(path, extra=False, type="gtCoarse"):
    """ return a list of paths to the cityscapes X and Y data

    Args:
        path (str): path to the dataset
        extra (bool, optional): if True, return the extra data. Defaults to False. (only gtCoarse)
        type (str, optional): type of the ground truth. Defaults to "gtCoarse".

    Returns:
        list: list of paths to the dataset in train+val, test split in the form (xTrainVal, yTrainVal, xTest, yTest)
    """
    print("Getting data from: " + path)

    path = join(path, "leftImg8bit", "__REPLACE__", "*", "*_leftImg8bit.png")

    xTrainVal = sorted(glob(path.replace("__REPLACE__", "train")))
    xTrainVal += sorted(glob(path.replace("__REPLACE__", "val")))

    print("Getting train data from: " + path.replace("__REPLACE__", "train"))
    print("Getting val data from: " + path.replace("__REPLACE__", "val"))

    if extra:
        xTrainVal += sorted(glob(path.replace("__REPLACE__", "train_extra")))
        print("Getting extra train data from: " + path.replace("__REPLACE__", "train_extra"))

    xTest = sorted(glob(path.replace("__REPLACE__", "test")))
    print("Getting test data from: " + path.replace("__REPLACE__", "test"))

    yTrainVal = getY(xTrainVal, type)
    yTest = getY(xTest, type)

    return xTrainVal, yTrainVal, xTest, yTest


def getY(data, type="gtCoarse"):
    """ return a list of paths to the data ground truth

    Args:
        data (list): list of paths to the data
        type (str, optional): type of the ground truth. Defaults to "gtCoarse".
    
    Returns:
        list: list of paths to the data ground truth in the form "dataPath" + "type" + "dataName"
    """
    yData = [i.replace("leftImg8bit", type) for i in data]
    return [i.replace("_" + type + ".png", "_" + type + "_labelIds.png") for i in yData]


def splitTrainValFolds(xData, yData, nSplits=5, seed=0):
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

    for trainIdx, valIdx in kf.split(xData):
        xTrain = [xData[i] for i in trainIdx]
        yTrain = [yData[i] for i in trainIdx]
        xVal = [xData[i] for i in valIdx]
        yVal = [yData[i] for i in valIdx]
        yield xTrain, yTrain, xVal, yVal

def dataShuffle(xData, yData, seed=0):
    """ Shuffle the data

    Args:
        xData (list): list of paths to the data
        yData (list): list of paths to the data ground truth
        seed (int, optional): seed for the random generator. Defaults to 0.

    Returns:
        list: list of shuffled data in the form (xData, yData)
    """
    random.seed(seed)
    toShuffle = list(zip(xData, yData))
    random.shuffle(toShuffle)
    return zip(*toShuffle)