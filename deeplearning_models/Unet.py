#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf


class CustomUNet3D(object):
    """
        Upgrade of version My_UNet3d_V10 in order to reduce overfitting

        custom Unet implementation for multiclasse semantic segmentation
    """

    def __init__(self,
                 image_shape,
                 number_class,
                 filters=(8, 16, 32, 64, 128),
                 kernel=(3, 3, 3),
                 activation=tf.keras.layers.LeakyReLU(),
                 padding='same',
                 pooling=(2, 2, 2)):

        self.num_classes = number_class
        self.image_shape = image_shape

        self.filters = filters
        self.kernel = kernel
        self.activation = activation
        self.padding = padding  # "same" or "valid"
        self.pooling = pooling

    def double_convolution(self, input_, num_filters):
        layer1 = tf.keras.layers.Conv3D(filters=num_filters,
                                        kernel_size=self.kernel,
                                        padding=self.padding
                                        )(input_)
        layer2 = self.activation(layer1)
        layer3 = tf.keras.layers.Conv3D(filters=num_filters,
                                        kernel_size=self.kernel,
                                        padding=self.padding
                                        )(layer2)
        layer4 = self.activation(layer3)
        layer5 = tf.keras.layers.SpatialDropout3D(0.2)(layer4)  # adapatation in progress
        return layer5

    def maxpooling(self, intput_):
        layer = tf.keras.layers.MaxPool3D(pool_size=self.pooling,
                                          padding=self.padding
                                          )(intput_)
        return layer

    def upsampling(self, input_):
        layer = tf.keras.layers.UpSampling3D(size=self.pooling
                                             )(input_)
        return layer

    @staticmethod
    def concatenate(upconv_input, forward_input):
        layer = tf.keras.layers.Concatenate()([upconv_input, forward_input])
        return layer

    def final_convolution(self, input_):
        layer = tf.keras.layers.Conv3D(filters=self.num_classes,
                                       kernel_size=(1, 1, 1),
                                       padding=self.padding
                                       )(input_)
        return layer

    def compression_block(self, input_, num_filters):
        """
            output : (forward_output, maxpooled_output)
        """
        layer1 = self.double_convolution(input_, num_filters)
        layer2 = self.maxpooling(layer1)
        layer3 = tf.keras.layers.BatchNormalization()(layer2)
        return (layer1, layer3)

    def bottleneck(self, input_, num_filters):
        layer1 = self.double_convolution(input_, num_filters)
        layer2 = tf.keras.layers.BatchNormalization()(layer1)
        return layer2

    def expansion_block(self, upconv_input, forward_input, num_filters):
        upconv_input = self.upsampling(upconv_input)
        layer1 = self.concatenate(upconv_input, forward_input)
        layer2 = self.double_convolution(layer1, num_filters)
        layer3 = tf.keras.layers.BatchNormalization()(layer2)
        return layer3

    def get_model(self):
        input_ = tf.keras.layers.Input(shape=self.image_shape, dtype=tf.float32, name="input")

        x = input_
        forwards = []

        # compression/encoder
        for i in range(len(self.filters) - 1):
            (forward, x) = self.compression_block(x, num_filters=self.filters[i])
            forwards.append(forward)

        # bottleneck
        x = self.bottleneck(x, num_filters=self.filters[-1])

        # expansion/decoder
        for i in reversed(range(len(self.filters) - 1)):
            x = self.expansion_block(x, forwards[i], num_filters=self.filters[i])

        # final layer
        logits = self.final_convolution(x)
        output_ = tf.keras.layers.Softmax(name='output')(logits)
        model = tf.keras.models.Model(input_, output_, name='UNet')

        return model
