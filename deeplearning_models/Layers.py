import tensorflow as tf


def get_spatial_rank(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the spatial rank of the tensor i.e. the number of spatial dimensions between batch_size and num_channels
    """
    return len(x.shape) - 2


def get_num_channels(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the number of channels of x
    """
    return int(x.shape[-1])


def get_spatial_size(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: The spatial shape of x, excluding batch_size and num_channels.
    """
    return x.shape[1:-1]

def prelu(x):
    return tf.keras.layers.PReLU(input_shape=x.get_shape()[-1])


def convolution(x, filters, kernel_size=(5, 5, 5), padding='same', strides=(1, 1, 1)):
    initializer = tf.keras.initializers.GlorotNormal()
    return tf.keras.layers.Conv3D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  strides=strides,
                                  kernel_initializer=initializer
                                  )(x)


def deconvolution(x, filters, kernel_size, strides, padding='same'):
    initializer = tf.keras.initializers.GlorotNormal()
    return tf.keras.layers.Conv3DTranspose(filters=filters,
                                           kernel_size=kernel_size,
                                           padding=padding,
                                           strides=strides,
                                           kernel_initializer=initializer
                                           )(x)


# More complex blocks


def down_convolution(x, factor, kernel_size):
    num_channels = get_num_channels(x)
    strides = 3 * [factor]
    filters = num_channels * factor
    x = convolution(x, filters, kernel_size=kernel_size, strides=strides)
    return x


def up_convolution(x, factor, kernel_size):
    num_channels = get_num_channels(x)
    strides = 3 * [factor]
    filters = num_channels // factor
    x = deconvolution(x, filters, kernel_size=kernel_size, strides=strides)
    return x
