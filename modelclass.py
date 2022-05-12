import tensorflow.keras.backend as K
import tensorflow as tf
import os
import datetime


# MODIFIED FROM https://github.com/Golbstein/Keras-segmentation-deeplab-v3.1/blob/master/utils.py
def sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_true = K.one_hot(
        tf.cast(y_true[:, :, :, 0], tf.int32), nb_classes)

    loss = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    loss = K.mean(loss, axis=[1, 2])
    return loss


def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = tf.cast(K.flatten(y_true), dtype=tf.int64)
    legal_labels = ~K.equal(y_true, nb_classes)
    return K.sum(tf.cast(legal_labels & K.equal(y_true, K.argmax(y_pred, axis=-1)),
                         dtype=tf.float32)) / K.sum(tf.cast(legal_labels, dtype=tf.float32))
# MODIFIED FROM https://github.com/Golbstein/Keras-segmentation-deeplab-v3.1/blob/master/utils.py


def dice_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_true = K.one_hot(
        tf.cast(y_true[:, :, :, 0], tf.int32), nb_classes)

    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)

    smooth = 1.0

    numerator = 2. * K.sum(y_true * y_pred, axis=-1) + smooth
    denominator = K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1) + smooth

    return numerator / denominator


def dice_loss_ignoring_last_label(y_true, y_pred):
    return 1 - dice_accuracy_ignoring_last_label(y_true, y_pred)


# Inspired by https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388
def sce_dice_loss(y_true, y_pred):
    dice_loss = dice_loss_ignoring_last_label(y_true, y_pred)
    sce_loss = sparse_crossentropy_ignoring_last_label(y_true, y_pred)

    return dice_loss + sce_loss


class MyMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1),
                                    tf.argmax(y_pred, axis=-1),
                                    sample_weight)


class Model():
    def __init__(self, image_size=256, num_classes=30, alpha=1., generate_model=None, path=None):
        self.image_size = image_size
        self.num_classes = num_classes
        self.alpha = alpha
        self.model = generate_model(
            image_size=self.image_size, num_classes=self.num_classes, alpha=self.alpha)
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
        loss_1 = sparse_crossentropy_ignoring_last_label
        loss_2 = dice_loss_ignoring_last_label
        loss_4 = sce_dice_loss

        accuracy_1 = {"sparse_accuracy": sparse_accuracy_ignoring_last_label}
        accuracy_2 = {"dice_accuracy": dice_accuracy_ignoring_last_label,
                      "sparse_accuracy": sparse_accuracy_ignoring_last_label}
        accuracy_4 = {"sparse_accuracy": sparse_accuracy_ignoring_last_label,
                      "dice_accuracy": dice_accuracy_ignoring_last_label}
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate, epsilon=epsilon, decay=decay),
            loss=loss_2,
            metrics=[accuracy_2],
            sample_weight_mode="temporal",
        )

    def train(self, x_train=None, x_val=None, batch_size=None, epochs=15,
              SAVE_PATH="save.hdf5", monitor="val_loss"):
        self.path_exists(SAVE_PATH)
        self.batch_size = batch_size
        callbacks = []

        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.path, monitor=monitor,
                                                        verbose=1, save_best_only=True, mode="auto")
        callbacks.append(checkpoint)

        earlyStopping = tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=100)
        callbacks.append(earlyStopping)

        reduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=20, verbose=1, min_lr=5e-10)
        callbacks.append(reduceLROnPlateau)

        # log_dir = "logs/fit/" + self.MODEL_NAME
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(
        #     log_dir=log_dir, histogram_freq=1)
        # callbacks.append(tensorboard_callback)

        self.history = self.model.fit(x=x_train, validation_data=x_val, batch_size=batch_size,
                                      epochs=epochs, callbacks=callbacks)

    def toTFlite(self, representative_dataset_gen=None, path=None,
                 supported_ops=[tf.lite.OpsSet.SELECT_TF_OPS,
                                tf.lite.OpsSet.TFLITE_BUILTINS,
                                tf.lite.OpsSet.TFLITE_BUILTINS_INT8]):

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
