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


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import Input

def deeplabV3(image_size=256, num_classes=20, alpha=1., with_argmax=False):
    """ Returns a model with MobineNetv2 backbone encoder and a DeeplabV3Plus decoder.
        Args:
            image_size - int - Input image size (image_size, image_size, channels)
            num_classes - int - Number of classes in output
            alpha - float - Alpha value for the MobileNetV2 backbone
            with_argmax - bool - If True, the model will output the argmax(image_size, image_size, classes), otherwise the reshaped output(image_size * image_size, num_classes)
    """
    def convolution_block(block_input, num_filters=256,
                          kernel_size=3, dilation_rate=1,
                          padding="same", use_bias=False,
                          kernel_regularizer=regularizers.L1L2(l1=0.01,
                                                               l2=0.001),
                          activity_regularizer=regularizers.L2(0),
                          activation=True, name=""):
        """ Returns a convolution block with the following structure:
        Args:
            block_input - Input tensor
            num_filters - int - Number of filters in the convolution
            kernel_size - int - Kernel size of the convolution
            dilation_rate - int - Dilation rate of the convolution
            padding - str - Padding of the convolution
            use_bias - bool - If True, a bias will be added to the convolution
            kernel_regularizer - Regularizer function applied to the kernel
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
            kernel_initializer=initializers.HeNormal(),
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            name=name + "_Conv2D"
        )(block_input)

        x = layers.BatchNormalization(epsilon=0.00001, name=name + "_BN")(x)

        if activation:
            x = layers.Activation('relu', name=name + "_ReLU")(x)

        return x

    def ASPP(aspp_input):
        """ Returns an ASPP block with the following structure:
            Args:
                aspp_input - Input tensor
            Returns:    
                x - Output tensor
        """
        dims = aspp_input.shape
        x = layers.AveragePooling2D(pool_size=(
            dims[-3], dims[-2]), name="assp_pool")(aspp_input)
        x = convolution_block(
            x, kernel_size=1, use_bias=True, name="aspp_pool")

        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
            interpolation="bilinear", name="aspp_pool_UpSampling"
        )(x)

        out_1 = convolution_block(
            aspp_input, kernel_size=1, dilation_rate=1, name="aspp_out1")

        x = layers.Concatenate(
            axis=-1, name="aspp_Concat")([out_pool, out_1])

        aspp_output = convolution_block(x, kernel_size=1, name="aspp_out")

        return layers.Dropout(.1, name="aspp_Dropout")(aspp_output)

    def DeeplabV3PlusMobileNetv2(image_size, num_classes):
        """ Returns a DeeplabV3Plus model with the following structure:
            Args:
                image_size - int - Input image size (image_size, image_size, channels)
                num_classes - int - Number of classes in output
            Returns:
                model - Keras model
        """
        model_input = Input(shape=(image_size, image_size, 3))
        mobilenetv2 = MobileNetV2(weights="imagenet", include_top=False,
                                  input_tensor=model_input, alpha=alpha)

        input_a = ASPP(mobilenetv2.get_layer("block_12_project_BN").output)
        input_b = mobilenetv2.get_layer("block_2_project_BN").output

        input_a = layers.UpSampling2D(
            size=(input_b.shape[1] // input_a.shape[1],
                  input_b.shape[2] // input_a.shape[2]),
            interpolation="bilinear", name="input_a_UpSampling"
        )(input_a)

        input_b = convolution_block(
            input_b, num_filters=48, kernel_size=1, name="input_b")

        x = layers.Concatenate(
            axis=-1, name="features_Concat")([input_a, input_b])

        x = layers.DepthwiseConv2D(
            kernel_size=3, padding="same", name="depthwise_separable_DepthwiseConv2D")(x)
        x = convolution_block(x, kernel_size=1, name="depthwise_separable_")
        x = layers.Dropout(.1, name="depthwise_separable_Dropout")(x)

        x = layers.Conv2D(num_classes, kernel_size=1,
                          padding="same", name="num_classes_Conv2D")(x)

        x = layers.UpSampling2D(
            size=(image_size // x.shape[1], image_size // x.shape[2]),
            interpolation="bilinear", name="output_UpSampling"
        )(x)

        x = layers.Activation(activation='softmax', name="softmax")(x)

        if with_argmax:
            x = layers.Lambda(lambda x: K.argmax(x, axis=-1), name="argmax")(x)
        else:
            x = layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]), name="output")(x)

        return Model(inputs=model_input, outputs=x, name="DeepLabV3Plus.py")

    return DeeplabV3PlusMobileNetv2(image_size, num_classes)


def deeplabV3_no_skip(image_size=256, num_classes=20, alpha=1., with_argmax=False):
    """ Returns a model with MobineNetv2 backbone encoder and a DeeplabV3Plus decoder.
    Args:
        image_size - int - Input image size (image_size, image_size, channels)
        num_classes - int - Number of classes in output
        alpha - float - Alpha value for the MobileNetV2 backbone
        with_argmax - bool - If True, the model will output the argmax(image_size, image_size, classes), otherwise the reshaped output(image_size * image_size, num_classes)
    """

    def convolution_block(block_input, num_filters=256,
                          kernel_size=3, dilation_rate=1,
                          padding="same", use_bias=False,
                          kernel_regularizer=regularizers.L1L2(l1=0.01,
                                                               l2=0.0001),
                          activity_regularizer=regularizers.L2(0),
                          activation=True, name=""):
        """ Returns a convolution block with the following structure:
        Args:
            block_input - Input tensor
            num_filters - int - Number of filters in the convolution
            kernel_size - int - Kernel size of the convolution
            dilation_rate - int - Dilation rate of the convolution
            padding - str - Padding of the convolution
            use_bias - bool - If True, a bias will be added to the convolution
            kernel_regularizer - Regularizer function applied to the kernel
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
            kernel_initializer=initializers.HeNormal(),
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            name=name + "_Conv2D"
        )(block_input)

        x = layers.BatchNormalization(epsilon=0.00001, name=name + "_BN")(x)

        if activation:
            x = layers.Activation('relu', name=name + "_ReLU")(x)

        return x

    def ASPP(aspp_input):
        """ Returns a ASPP block with the following structure:
            Args:
                aspp_input - Input tensor
            Returns:
                x - Output tensor
        """
        dims = aspp_input.shape
        x = layers.AveragePooling2D(pool_size=(
            dims[-3], dims[-2]), name="assp_pool")(aspp_input)
        x = convolution_block(
            x, kernel_size=1, use_bias=True, name="aspp_pool")

        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
            interpolation="bilinear", name="aspp_pool_UpSampling"
        )(x)

        out_1 = convolution_block(
            aspp_input, kernel_size=1, dilation_rate=1, name="aspp_out1")

        x = layers.Concatenate(
            axis=-1, name="aspp_Concat")([out_pool, out_1])

        aspp_output = convolution_block(x, kernel_size=1, name="aspp_out")

        return layers.Dropout(.1, name="aspp_Dropout")(aspp_output)

    def DeeplabV3PlusMobileNetv2(image_size, num_classes):
        """ Returns a DeeplabV3Plus model with the following structure:
            Args:
                image_size - int - Input image size (image_size, image_size, channels)
                num_classes - int - Number of classes in output
            Returns:
                model - Keras model
        """
        model_input = Input(shape=(image_size, image_size, 3))
        mobilenetv2 = MobileNetV2(weights="imagenet", include_top=False,
                                  input_tensor=model_input, alpha=alpha)

        x = ASPP(mobilenetv2.get_layer("block_12_project_BN").output)

        x = layers.DepthwiseConv2D(
            kernel_size=3, padding="same", name="depthwise_separable_DepthwiseConv2D")(x)
        x = convolution_block(x, kernel_size=1, name="depthwise_separable_")
        x = layers.Dropout(.1, name="depthwise_separable_Dropout")(x)

        x = layers.Conv2D(num_classes, kernel_size=1,
                          padding="same", name="num_classes_Conv2D")(x)

        x = layers.UpSampling2D(
            size=(image_size // x.shape[1], image_size // x.shape[2]),
            interpolation="bilinear", name="output_UpSampling"
        )(x)

        x = layers.Activation('softmax', name="softmax")(x)

        if with_argmax:
            x = layers.Lambda(lambda x: K.argmax(x, axis=-1), name="argmax")(x)
        else:
            x = layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]), name="output")(x)

        return Model(inputs=model_input, outputs=x, name="DeepLabV3Plus.py")

    return DeeplabV3PlusMobileNetv2(image_size, num_classes)