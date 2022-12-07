from lossesAccuracyfuncs import lossesAccuracyClass
import tensorflow.keras.backend as K
import tensorflow as tf
import models
import os
import datetime
import numpy as np

class DeeplabV3(tf.keras.Model):
    def __init__(self, imageSize=256, nClasses=30, alpha=1., path="model-saved", modelName="deeplabV3MobileNetV2", withArgmax=False, noSkip=False):
        """ Initialize the model
            args:
                imageSize: image size
                nClasses: number of classes
                alpha: alpha for MobileNetV2 backbone
                generate_model: generate model function
                path: path to load/save the model
                modelName: name of the model
                withArgmax: whether to use argmax to generate the mask
                noSkip: whether to use skip connections
        """
        super(DeeplabV3, self).__init__()

        self.imageSize = imageSize
        self.nClasses = nClasses
        self.alpha = alpha
        self.modelName = os.path.join(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), modelName)
        self.path = os.path.join(path, modelName)
        self.model = None
        self.withArgmax = withArgmax
        self.noSkip = noSkip


        if self.noSkip:
            self.model = models.deeplabV3(imageSize=self.imageSize, nClasses=self.nClasses,
                                          alpha=self.alpha, withArgmax=withArgmax)
        else:
            self.model = models.deeplabV3_no_skip(imageSize=self.imageSize, nClasses=self.nClasses,
                                                  alpha=self.alpha, withArgmax=withArgmax)

    def call(self, inputs):
        return self.model(inputs)

    def summary(self):
        self.model.summary()

    def load_weights(self, path=None):
        if path is not None:
            self.model.load_weights(path)
        else:
            self.model.load_weights(self.path)
        
    def save_weights(self, path=None):
        if path is not None:
            self.model.save_weights(path)
        else:
            self.model.save_weights(self.path)

    def __loss__(self):
        return lossesAccuracyClass().diceLoss
    
    def __metrics__(self):
        # return [lossesAccuracyClass().accuracy, lossesAccuracyClass().precision, lossesAccuracyClass().recall,
        #         lossesAccuracyClass().f1Score, lossesAccuracyClass().mIOU, lossesAccuracyClass().jaccardDistance,
        #         lossesAccuracyClass().diceAccuracy]
        return [lossesAccuracyClass().diceAccuracy, lossesAccuracyClass().jaccardDistance]

    def compile(self, learningRate=None):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
            loss=self.__loss__(),
            metrics=self.__metrics__(),
            # run_eagerly=True,
            sample_weight_mode="temporal",
        )

    def __callbacks__(self, monitor=None, log=False):
        callbacksList = []

        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.path, monitor=monitor,
                                                        verbose=1, save_best_only=True,
                                                        mode="auto")
        callbacksList.append(checkpoint)

        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=1000)
        callbacksList.append(earlyStopping)

        reduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                                                 patience=100, verbose=0,
                                                                 min_lr=5e-15)
        callbacksList.append(reduceLROnPlateau)

        if log:
            log_dir = "logs/fit/" + self.modelName
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            callbacksList.append(tensorboard_callback)

        return callbacksList

    def run(self, xData, valData, batchSize=4, learningRate=1e-2, epochs=100, monitor="mIOU", log=False):
        self.compile(learningRate=learningRate)
        
        self.model.fit(x=xData, validation_data=valData, batch_size=batchSize,
                       epochs=epochs, callbacks=self.__callbacks__(monitor=monitor, log=log))


    def toTFlite(self, representativeDatasetGen=None,
                    supportedOps=[tf.lite.OpsSet.SELECT_TF_OPS,
                                  tf.lite.OpsSet.TFLITE_BUILTINS,
                                  tf.lite.OpsSet.TFLITE_BUILTINS_INT8]):

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representativeDatasetGen
        converter.target_spec.supported_ops = supportedOps
        tflite_model = converter.convert()

        with open(self.path + ".tflite", "wb") as f:
            f.write(tflite_model)
