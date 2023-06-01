# -*- coding: utf-8 -*-

""" Deeplabv3+ model for Keras.
This model is based on the example:
https://keras.io/examples/vision/deeplabv3_plus/
# Reference
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)
"""

from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import Input


def deeplabV3(imageSize=(256, 256), nClasses=20, alpha=1., withArgmax=False,
              mobileLayers=None, kernel_regularizer=None, SEED=None,
              kernel_initializer=None, frozen=False):
    """ Returns a model with MobineNetv2 backbone encoder and a DeeplabV3Plus decoder.
        Args:
            imageSize - int - Input image size (imageSize, imageSize, channels)
            nClasses - int - Number of classes in output
            alpha - float - Alpha value for the MobileNetV2 backbone
            withArgmax - bool - If True, the model will output the argmax(imageSize, imageSize, classes), otherwise the reshaped output(imageSize * imageSize, nClasses)
    """
    if kernel_initializer is None:
        kernel_initializer = initializers.RandomUniform(minval=-2.5, maxval=2.5, seed=SEED)
        
    if mobileLayers is None:
        mobileLayers = {"shallowLayer": "block_2_project_BN", 
                        "deepLayer": "block_12_project_BN"}

    def convolution_block(block_input, num_filters=256, kernel_size=3,
                          dilation_rate=1, padding="same", use_bias=False,
                          activation=True, activation_type_in="relu", name=""):
        """ Returns a convolution block with the following structure:
        Args:
            block_input - Input tensor
            num_filters - int - Number of filters in the convolution
            kernel_size - int - Kernel size of the convolution
            dilation_rate - int - Dilation rate of the convolution
            padding - str - Padding of the convolution
            use_bias - bool - If True, a bias will be added to the convolution
            activity_regularizer - Regularizer function applied to the activity
            activation - bool - If True, a relu activation will be applied
            name - str - Name of the block
        Returns:
            x - Output tensor
        """
        x = layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name + "_Conv2D"
        )(block_input)

        x = layers.BatchNormalization(epsilon=0.00001, name=name + "_BN")(x)

        if activation:
            x = layers.Activation(activation_type_in, name=name + "_ReLU")(x)

        return x

    def SeparableConv2D_block(block_input, num_filters=256, kernel_size=3,
                              dilation_rate=1, padding="same", use_bias=False,
                              activation=True, activation_type_in="relu", name=""):
        """ Returns a convolution block with the following structure:
        Args:
            block_input - Input tensor
            num_filters - int - Number of filters in the convolution
            kernel_size - int - Kernel size of the convolution
            dilation_rate - int - Dilation rate of the convolution
            padding - str - Padding of the convolution
            use_bias - bool - If True, a bias will be added to the convolution
            activity_regularizer - Regularizer function applied to the activity
            activation - bool - If True, a relu activation will be applied
            name - str - Name of the block
        Returns:
            x - Output tensor
        """
        x = layers.SeparableConv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name + "_SeparableConv2D"
        )(block_input)

        x = layers.BatchNormalization(epsilon=0.00001, name=name + "_BN")(x)

        if activation:
            x = layers.Activation(activation_type_in, name=name + "_ReLU")(x)

        return x


    def ASPP(aspp_input):
        """ Returns an ASPP block with the following structure:
            Args:
                aspp_input - Input tensor
            Returns:    
                x - Output tensor
        """
        dims = aspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]),
                                    name="assp_pool")(aspp_input)
        x = convolution_block(x, kernel_size=1, use_bias=True,
                              name="aspp_pool")

        out_pool = layers.UpSampling2D(size=(dims[-3] // x.shape[1],
                                             dims[-2] // x.shape[2]),
            interpolation="bilinear", name="aspp_pool_UpSampling")(x)

        out_1 = convolution_block(
            aspp_input, kernel_size=1, dilation_rate=1, name="aspp_out1")

        x = layers.Concatenate(axis=-1, name="aspp_Concat")([out_pool, out_1])

        aspp_output = convolution_block(x, kernel_size=1, name="aspp_out")

        return layers.Dropout(.25, name="aspp_Dropout")(aspp_output)

    def DeeplabV3PlusMobileNetv2(imageSize, nClasses):
        """ Returns a DeeplabV3Plus model with the following structure:
            Args:
                imageSize - int - Input image size (imageSize, imageSize, channels)
                nClasses - int - Number of classes in output
            Returns:
                model - Keras model
        """
        model_input = Input(shape=(imageSize[0], imageSize[1], 3))
        model_input = layers.Rescaling(scale=1./127.5, offset=-1)(model_input)
        mobilenetv2 = MobileNetV2(weights="imagenet", include_top=False,
                                  input_tensor=model_input, alpha=alpha)

        if frozen:
            for layer in mobilenetv2.layers:
                layer.trainable = False

        input_a = ASPP(mobilenetv2.get_layer(mobileLayers["deepLayer"]).output)
        input_b = mobilenetv2.get_layer(mobileLayers["shallowLayer"]).output

        input_a = layers.UpSampling2D(size = (4, 4),
                                      interpolation="bilinear",
                                      name="input_a_UpSampling")(input_a)

        input_b = convolution_block(input_b, num_filters=48, kernel_size=1,
                                    name="input_b")

        x = layers.Concatenate(axis=-1, name="features_Concat")([input_a, input_b])
    
        x = SeparableConv2D_block(x, kernel_size=3, name="output_")

        x = layers.UpSampling2D(size = (4, 4), interpolation="bilinear",
                                name="output_UpSampling")(x)

        x = layers.Conv2D(nClasses, kernel_size=1, padding="same",
                          name="nClasses_Conv2D")(x)

        x = layers.Activation(activation='softmax', name="softmax")(x)

        if withArgmax:
            x = layers.Lambda(lambda x : K.argmax(x, axis=-1), name="argmax")(x)
        else:
            x = layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]),
                               name="output")(x)

        return Model(inputs=model_input, outputs=x, name="DeepLabV3Plus")

    if isinstance(imageSize, int):
        imageSize = (imageSize, imageSize)

    return DeeplabV3PlusMobileNetv2(imageSize=imageSize, nClasses=nClasses)