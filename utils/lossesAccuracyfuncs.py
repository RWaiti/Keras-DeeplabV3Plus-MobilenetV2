from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

class Losses_n_Metrics():
    def __init__(self):
        pass

    def diceAccuracy(self, y_true, y_pred, smooth=1):
        """ Dice loss - Ignore last class from true mask
            args:
                y_true: ground truth 4D keras tensor (B,H,W,C)
                y_pred: predicted 4D keras tensor (B,H,W,C)
                smooth: value to avoid division by zero
            return:
                dice loss
        """

        nb_classes = K.int_shape(y_pred)[-1]

        y_true_f = K.one_hot(
            tf.cast(y_true[:, :, 0], tf.int32), nb_classes + 1)[:, :, :-1]

        y_true_f = K.batch_flatten(y_true_f)
        y_pred_f = K.batch_flatten(y_pred)

        intersection = 2. * K.sum(y_true_f * y_pred_f, axis=-1) + smooth
        union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1) + smooth

        return intersection / union


    def diceLoss(self, y_true, y_pred):
        """ Dice loss 
            args:
                y_true: ground truth 4D keras tensor (B,H,W,C)
                y_pred: predicted 4D keras tensor (B,H,W,C) 
            return:
                dice loss
        """
        return 1 - self.diceAccuracy(y_true, y_pred)


    def jaccardDistance(self, y_true, y_pred, smooth=10):
        """ Jaccard distance
            args:
                y_true: ground truth 4D keras tensor (B,H,W,C)
                y_pred: predicted 4D keras tensor (B,H,W,C)
                smooth: value to avoid division by zero
            return:
                jaccard distance
        """
        nb_classes = K.int_shape(y_pred)[-1]
        y_true_f = K.one_hot(
            tf.cast(y_true[:, :, 0], tf.int32), nb_classes + 1)[:, :, :-1]

        y_true_f = K.batch_flatten(y_true_f)
        y_pred_f = K.batch_flatten(y_pred)

        intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1) + smooth
        sum_ = K.sum(K.abs(y_true_f) + K.abs(y_pred_f), axis=-1) + smooth

        jac = intersection / (sum_ - intersection)

        return (1 - jac) * smooth

    def accuracy(self, y_true, y_pred):
        """ Accuracy = True Positive + True Negative / (True Positive + False Positive + True Negative + False Negative)
            args:
                y_true: ground truth 4D keras tensor (B,H,W,C)
                y_pred: predicted 4D keras tensor (B,H,W,C)
                return:
                    accuracy
        """
        y_pred_numpy = np.argmax(y_pred.numpy(), axis=-1)
        y_true_numpy = y_true[:, :, 0].numpy()

        batch_size = y_pred_numpy.shape[0]
        accuracy_ = 0

        for i in range(batch_size):
            # True positive 1 & 1
            TP = np.bitwise_and(
                (y_true_numpy[i] == 1), (y_pred_numpy[i] == 1)).sum()
            # False positive 0 & 1
            FP = np.bitwise_and(
                (y_true_numpy[i] == 0), (y_pred_numpy[i] == 1)).sum()
            # True negative 0 & 0
            TN = np.bitwise_and(
                (y_true_numpy[i] == 0), (y_pred_numpy[i] == 0)).sum()
            # False negative 1 & 0
            FN = np.bitwise_and(
                (y_true_numpy[i] == 1), (y_pred_numpy[i] == 0)).sum()

            accuracy_ += (TP + TN) / (TP + FP + TN + FN + K.epsilon())

        return accuracy_ / batch_size


    def precision(self, y_true, y_pred):
        """ Precision = True Positive / (True Positive + False Positive)))
            args:
                y_true: ground truth 4D keras tensor (B,H,W,C)
                y_pred: predicted 4D keras tensor (B,H,W,C)
            return:
                precision
        """
        y_pred_numpy = np.argmax(y_pred.numpy(), axis=-1)
        y_true_numpy = y_true[:, :, 0].numpy()

        batch_size = y_pred_numpy.shape[0]
        precision_ = 0

        for i in range(batch_size):
            # True positive 1 & 1
            TP = np.bitwise_and(
                (y_true_numpy[i] == 1), (y_pred_numpy[i] == 1)).sum()
            # False positive 0 & 1
            FP = np.bitwise_and(
                (y_true_numpy[i] == 0), (y_pred_numpy[i] == 1)).sum()

            precision_ += (TP) / (TP + FP + K.epsilon())

        return precision_ / batch_size


    def recall(self, y_true, y_pred):
        """ Recall =  True Positive / (True Positive + False Negative) 
            args:
                y_true: ground truth 4D keras tensor (B,H,W,C)
                y_pred: predicted 4D keras tensor (B,H,W,C)
            return:
                recall
        """
        y_pred_numpy = np.argmax(y_pred.numpy(), axis=-1)
        y_true_numpy = y_true[:, :, 0].numpy()

        batch_size = y_pred_numpy.shape[0]
        recall_ = 0

        for i in range(batch_size):
            # True positive - 1 & 1
            TP = np.bitwise_and(
                (y_true_numpy[i] == 1), (y_pred_numpy[i] == 1)).sum()
            # False negative - 1 & 0
            FN = np.bitwise_and(
                (y_true_numpy[i] == 1), (y_pred_numpy[i] == 0)).sum()

            recall_ += (TP) / (TP + FN + K.epsilon())

        return recall_ / batch_size


    def f1Score(self, y_true, y_pred):
        """ F1 Score = 2 * Precision * Recall / (Precision + Recall)
            args:
                y_true: ground truth 4D keras tensor (B,H,W,C)
                y_pred: predicted 4D keras tensor (B,H,W,C)
            return:
                f1 score
        """
        y_pred_numpy = np.argmax(y_pred.numpy(), axis=-1)
        y_true_numpy = y_true[:, :, 0].numpy()

        batch_size = y_pred_numpy.shape[0]

        precision_ = np.zeros(batch_size)
        recall_ = np.zeros(batch_size)

        for i in range(batch_size):
            TP = np.bitwise_and(
                (y_true_numpy[i] == 1), (y_pred_numpy[i] == 1)).sum()
            FP = np.bitwise_and(
                (y_true_numpy[i] == 0), (y_pred_numpy[i] == 1)).sum()
            FN = np.bitwise_and(
                (y_true_numpy[i] == 1), (y_pred_numpy[i] == 0)).sum()

            precision_[i] = (TP) / (TP + FP + K.epsilon())
            recall_[i] = (TP) / (TP + FN + K.epsilon())

        f1Score_ = 2 * ((precision_ * recall_) /
                        (precision_ + recall_ + K.epsilon()))

        return f1Score_.mean()


    def mIOU(self, y_true, y_pred):
        """ Mean Intersection over Union
            args:
                y_true: ground truth 4D keras tensor (B,H,W,C)
                y_pred: predicted 4D keras tensor (B,H,W,C)
            return:
                mIOU
        """
        y_pred_numpy = np.argmax(y_pred.numpy(), axis=-1)
        y_true_numpy = y_true[:, :, 0].numpy()

        batch_size = y_pred_numpy.shape[0]
        ulabels = np.unique(y_true_numpy).astype(np.uint8)
        iou = np.zeros(batch_size)

        for i in range(batch_size):
            iou_temp = np.zeros(len(ulabels))

            for k, u in enumerate(ulabels):
                inter = (y_true_numpy == u) & (y_pred_numpy == u)
                union = (y_true_numpy == u) | (y_pred_numpy == u)

                iou_temp[k] = inter.sum() / union.sum()

            iou[i] = iou_temp.mean()

        return iou.mean()