from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

# KERAS EXAMPLE : https://io/examples/vision/deeplabv3_plus/


def generate_model(image_size=256, num_classes=20, alpha=1.):
    '''
    Returns a model with MobineNetv2 or EfficientNetB0 backbone encoder and a DeeplabV3Plus decoder.
    image_size - Input image size
        image_size=256(default)
    num_classes - int - Output number classes
        num_classes=20(default)
    backbone - Backbone Architecture
        name="MobileNetV2" (default)
    '''
    # FROM KERAS EXAMPLE
    def convolution_block(block_input, num_filters=256,
                          kernel_size=3, dilation_rate=1,
                          padding="same", use_bias=False,
                          activation=True, kernel_regularizer=None):
        x = layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=initializers.LecunUniform(),
            kernel_regularizer=kernel_regularizer,
        )(block_input)

        x = layers.BatchNormalization(epsilon=0.00001)(x)

        if activation:
            return layers.ReLU(max_value=6.0)(x)
        return x
        # QAT made me modify from tf.nn.relu(X)
        # tensorflow_model_optimization.quantization.keras.quantize_model()
        # only supports keras layers
    # FROM KERAS EXAMPLE

    def ASPP(aspp_input):
        dims = aspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(aspp_input)
        x = convolution_block(x, kernel_size=1, use_bias=True)

        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
            interpolation="bilinear"
        )(x)

        out_1 = convolution_block(aspp_input, kernel_size=1, dilation_rate=1)
        out_6 = convolution_block(aspp_input, kernel_size=3, dilation_rate=6)
        out_12 = convolution_block(aspp_input, kernel_size=3, dilation_rate=12)
        out_18 = convolution_block(aspp_input, kernel_size=3, dilation_rate=18)

        # x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6])
        # x = layers.Concatenate(axis=-1)([out_pool, out_1])

        return convolution_block(x, kernel_size=1)

    # BASED ON THE KERAS EXAMPLE, USING ANOTHER ARCHITECTURE AND ADDING ACTIVATION LAYER
    def DeeplabV3PlusMobileNetv2(image_size, num_classes):
        model_input = layers.Input(shape=(image_size, image_size, 3))
        mobilenetv2 = MobileNetV2(weights="imagenet", include_top=False,
                                  input_tensor=model_input, alpha=alpha)

        get_layer_1 = "block_12_project_BN"
        get_layer_2 = "block_2_project_BN"

        x = mobilenetv2.get_layer(get_layer_1).output
        x = ASPP(x)
        x = layers.Dropout(.1)(x)

        input_a = layers.UpSampling2D(
            size=(image_size // 4 // x.shape[1],
                  image_size // 4 // x.shape[2]),
            interpolation='bilinear',
        )(x)

        input_b = mobilenetv2.get_layer(get_layer_2).output
        input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

        x = layers.Concatenate(axis=-1)([input_a, input_b])

        x = convolution_block(x, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))
        x = layers.Dropout(.1)(x)
        x = convolution_block(x, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))

        x = layers.UpSampling2D(
            size=(image_size // x.shape[1], image_size // x.shape[2]),
            interpolation='bilinear',
        )(x)

        x = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)

        model_output = layers.Activation('softmax', name='output_mask')(x)

        return Model(inputs=model_input, outputs=model_output)

    # BASED ON THE KERAS EXAMPLE, USING ANOTHER ARCHITECTURE AND ADDING ACTIVATION LAYER
    return DeeplabV3PlusMobileNetv2(image_size, num_classes)


def generate_model_no_skip(image_size=256, num_classes=20, alpha=1.):
    '''
    Returns a model with MobineNetv2 or EfficientNetB0 backbone encoder and a DeeplabV3Plus decoder.
    image_size - Input image size
        image_size=256(default)
    num_classes - int - Output number classes
        num_classes=20(default)
    backbone - Backbone Architecture
        name="MobileNetV2" (default)
    '''
    # FROM KERAS EXAMPLE
    def convolution_block(block_input, num_filters=256,
                          kernel_size=3, dilation_rate=1,
                          padding="same", use_bias=False,
                          activation=True, kernel_regularizer=None):
        x = layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=initializers.LecunUniform(),
            kernel_regularizer=kernel_regularizer,
        )(block_input)

        x = layers.BatchNormalization(epsilon=0.00001)(x)

        if activation:
            return layers.ReLU(max_value=6.0)(x)
        return x
        # QAT made me modify from tf.nn.relu(X)
        # tensorflow_model_optimization.quantization.keras.quantize_model()
        # only supports keras layers
    # FROM KERAS EXAMPLE

    def ASPP(aspp_input):
        dims = aspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(aspp_input)
        x = convolution_block(x, kernel_size=1, use_bias=True)

        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
            interpolation="bilinear"
        )(x)

        out_1 = convolution_block(aspp_input, kernel_size=1, dilation_rate=1)
        out_6 = convolution_block(aspp_input, kernel_size=3, dilation_rate=6)
        out_12 = convolution_block(aspp_input, kernel_size=3, dilation_rate=12)
        out_18 = convolution_block(aspp_input, kernel_size=3, dilation_rate=18)

        # x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6])
        # x = layers.Concatenate(axis=-1)([out_pool, out_1])

        return convolution_block(x, kernel_size=1)

    # BASED ON THE KERAS EXAMPLE, USING ANOTHER ARCHITECTURE AND ADDING ACTIVATION LAYER
    def DeeplabV3PlusMobileNetv2(image_size, num_classes):
        model_input = layers.Input(shape=(image_size, image_size, 3))
        mobilenetv2 = MobileNetV2(weights="imagenet", include_top=False,
                                  input_tensor=model_input, alpha=alpha)

        x = mobilenetv2.get_layer("block_12_project_BN").output
        x = ASPP(x)
        x = layers.Dropout(.1)(x)

        x = convolution_block(x)
        x = layers.Dropout(.1)(x)
        x = convolution_block(x)

        x = layers.UpSampling2D(size=(image_size // x.shape[1],
                                      image_size // x.shape[2]),
                                interpolation='bilinear')(x)

        x = layers.Conv2D(num_classes, kernel_size=1, padding="same")(x)

        model_output = layers.Activation('softmax', name="output_mask")(x)

        return Model(inputs=model_input, outputs=model_output)

    # BASED ON THE KERAS EXAMPLE, USING ANOTHER ARCHITECTURE AND ADDING ACTIVATION LAYER
    return DeeplabV3PlusMobileNetv2(image_size, num_classes)
