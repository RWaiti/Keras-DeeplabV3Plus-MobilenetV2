import tensorflow.keras.backend as K
import tensorflow as tf
import os
import datetime
import numpy as np


def dice_accuracy(y_true, y_pred, smooth=1):
    """ Dice loss
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


def dice_loss(y_true, y_pred):
    """ Dice loss 
        args:
            y_true: ground truth 4D keras tensor (B,H,W,C)
            y_pred: predicted 4D keras tensor (B,H,W,C) 
        return:
            dice loss
    """
    return 1 - dice_accuracy(y_true, y_pred)


def jaccardDistance(y_true, y_pred, smooth=10):
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


def accuracy(y_true, y_pred):
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


def precision(y_true, y_pred):
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


def recall(y_true, y_pred):
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


def f1Score(y_true, y_pred):
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


def mIOU(y_true, y_pred):
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



class Model():
    def __init__(self, image_size=256, num_classes=30, alpha=1.,
                 generate_model=None, path=None, with_argmax=False):
        """ Initialize the model
            args:
                image_size: image size
                num_classes: number of classes
                alpha: alpha for MobileNetV2 backbone
                generate_model: generate model function
                path: path to load/save the model
                with_argmax: whether to use argmax to generate the mask
        """
        self.image_size = image_size
        self.num_classes = num_classes
        self.alpha = alpha
        self.model = generate_model(image_size=self.image_size, num_classes=self.num_classes,
                                    alpha=self.alpha, with_argmax=with_argmax)
        self.path = path

    def summary(self):
        self.model.summary()

    def path_exists(self, path):
        if path is not None:
            self.path = path
        elif self.path is None:
            raise Exception(
                "path and self.path are both NoneType, please insert a path")

    def load_weights(self, path=None):
        self.path_exists(path)
        self.model.load_weights(self.path)

    def compile(self, learning_rate=7e-4, epsilon=1e-8, decay=1e-6):
        loss = dice_loss  # binary

        accuracy = [accuracy, precision, recall,
                      f1Score, mIOU, jaccardDistance,
                      dice_accuracy]

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=accuracy,
            run_eagerly=True,
            sample_weight_mode="temporal",
        )

    def train(self, x_train=None, x_val=None, batch_size=None, epochs=None,
              SAVE_PATH=None, monitor="val_loss"):
        self.path_exists(SAVE_PATH)
        self.batch_size = batch_size
        callbacks = []

        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.path, monitor=monitor,
                                                        verbose=1, save_best_only=True,
                                                        mode="auto")
        callbacks.append(checkpoint)

        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=50)
        callbacks.append(earlyStopping)

        reduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                                                 patience=10, verbose=0,
                                                                 min_lr=5e-10)
        callbacks.append(reduceLROnPlateau)

        # log_dir = "logs/fit/" + self.MODEL_NAME
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(
        #     log_dir=log_dir, histogram_freq=1)
        # callbacks.append(tensorboard_callback)

        self.history = self.model.fit(x=x_train, validation_data=x_val, batch_size=batch_size,
                                      epochs=epochs, callbacks=callbacks, workers=1, verbose=1)

    def toTFlite(self, representative_dataset_gen=None, path=None,
                 supported_ops=[tf.lite.OpsSet.SELECT_TF_OPS,
                                tf.lite.OpsSet.TFLITE_BUILTINS,
                                tf.lite.OpsSet.TFLITE_BUILTINS_INT8]):
        """ Convert the model to TFlite format
            args:
                representative_dataset_gen: function to generate representative dataset
                path: path to save the model
                supported_ops: supported ops
        """

        def converterLite(supported_ops=[]):
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = supported_ops
            return converter.convert()

        def saveQuantized(quantized_tflite_model=None,
                          GENERAL_PATH=None, MODEL_NAME=None):
            if None in [quantized_tflite_model, GENERAL_PATH, MODEL_NAME]:
                raise Exception("Need all arguments")

            save_path = os.path.join(GENERAL_PATH, MODEL_NAME + ".tflite")

            open(save_path, "wb").write(quantized_tflite_model)
            print("Modelo Salvo em " + save_path)

        self.path_exists(path)
        if representative_dataset_gen is None:
            raise Exception("representative_dataset_gen can't be NoneType")

        for letter in range(len(self.path) - 1, -1, -1):
            if self.path[letter] == "/":
                save_position = letter
                break

        GENERAL_PATH = self.path[:save_position]

        for i, operations in enumerate(supported_ops):
            MODEL_NAME = self.path[save_position + 1:]

            print(i + 1, "of", len(supported_ops), ":", operations.name)

            quantized_tflite_model = converterLite(
                supported_ops=[operations])

            MODEL_NAME = operations.name + "_" + MODEL_NAME[:-5]

            saveQuantized(quantized_tflite_model=quantized_tflite_model,
                          GENERAL_PATH=GENERAL_PATH, MODEL_NAME=MODEL_NAME)
