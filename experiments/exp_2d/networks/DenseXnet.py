import tensorflow as tf
from networks.Layers import prelu


def BatchNorm(x, training=True):
    # print(x.shape)
    return tf.keras.layers.BatchNormalization()(x, training=training)


def get_num_channels(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the number of channels of x
    """
    return int(x.shape[-1])


def deconvolution(x, filters, kernel_size, strides, padding='same'):
    return tf.keras.layers.Conv2DTranspose(filters=filters,
                                           kernel_size=kernel_size,
                                           padding=padding,
                                           strides=strides
                                           )(x)


def up_convolution(x, factor, kernel_size):
    num_channels = get_num_channels(x)
    strides = 2 * [factor]
    filters = num_channels // factor
    x = deconvolution(x, filters, kernel_size=kernel_size, strides=strides)
    return x


def dense_block_2d(x, filters, num_conv, kernel_size):
    """
    :param x: input of shape m x m @c
    :param filters: int, inner
    :param num_conv: int, number of convolution
    :param kernel_size: int or tuple
    :return: output of shape m x m @(c + filters * num_conv)
    """
    for i in range(num_conv):
        # print(x.shape)
        y = BatchNorm(x)
        # print(y.shape)
        y = tf.keras.layers.ReLU()(y)
        y = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   padding='same'
                                   )(y)
        x = tf.keras.layers.Concatenate()([x, y])

    return x


def transition_down(x):
    return tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)


def transition_up(x, kernel_size, factor=2):
    x = BatchNorm(x)
    x = tf.keras.layers.ReLU()(x)
    # filters = get_num_channels(x) // 2
    # x = tf.keras.layers.Conv2D(filters=filters,
    #                            kernel_size=1,
    #                            padding='same'
    #                            )(x)
    # x = tf.keras.layers.ReLU()(x)
    x = up_convolution(x, factor, kernel_size)
    return x


def reduce_features_map(x, factor=2):
    filters = get_num_channels(x) // factor
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=1,
                               padding='same'
                               )(x)
    return tf.keras.layers.ReLU()(x)


class DenseXNet(object):
    """
    Implements DenseX-Net  architecture https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8946601
    """

    def __init__(self,
                 input_shape, in_channels, out_channels,
                 num_levels=5, kernel_size=3, filters_block=8, num_conv_block=8, activation_fn='PRELU'):
        """
        :param input_shape:      tuple, shape of the input without channel axis.
        :param in_channels:      int, number of input channels
        :param out_channels:     int,  number of output channels
        :param num_levels:       int, The number of levels in the network. Default is 5 as in the paper.
        :param kernel_size:      tuple or int, convolution kernel size
        :param filters_block     int, the number of filters in the dense block
        :param num_conv_block    int, the number of convolution per dense block
        :param activation_fn:    str or tf activation fc .The activation function or its name.
        """

        self.image_shape = tuple(list(input_shape) + [in_channels])  # channel last
        # self.dimensions = len(input_shape)
        # assert (self.dimensions == 2 or self.dimensions == 3)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.filters = filters_block
        self.num_conv = num_conv_block
        if isinstance(activation_fn, str):
            self.activation_fn = prelu if activation_fn.lower() == 'prelu' else tf.keras.activations.get(activation_fn)
        else:
            self.activation_fn = activation_fn

    def build_network(self, input_):
        x = input_

        x = tf.keras.layers.Conv2D(filters=16,
                                   kernel_size=self.kernel_size,
                                   padding='same'
                                   )(x)
        forwards = list()
        for i in range(self.num_levels):
            if i == 0:
                x = dense_block_2d(x, filters=self.filters, num_conv=self.num_conv, kernel_size=self.kernel_size)
                forwards.append(x)
            else:
                x = transition_down(x)
                x = dense_block_2d(x, filters=self.filters, num_conv=self.num_conv, kernel_size=self.kernel_size)
                forwards.append(x)

        x = transition_down(x)
        x = dense_block_2d(x, filters=self.filters, num_conv=self.num_conv, kernel_size=self.kernel_size)
        # reconstruction = x

        for i in reversed(range(self.num_levels)):
            x = transition_up(x, self.kernel_size, 2)
            x = dense_block_2d(x, filters=self.filters, num_conv=self.num_conv, kernel_size=self.kernel_size)
            x = tf.keras.layers.Concatenate()([x, forwards[i]])

        logits_seg = tf.keras.layers.Conv2D(filters=1,
                                            kernel_size=self.out_channels,
                                            padding='same'
                                            )(x)

        # # encoder to reconstruction target
        # for i in reversed(range(self.num_levels)):
        #     reconstruction = transition_up(reconstruction, self.kernel_size, 2)
        #     reconstruction = dense_block_2d(reconstruction, filters=self.filters, num_conv=self.num_conv,
        #                                       kernel_size=self.kernel_size)

        # reconstruction = tf.keras.layers.Conv2D(filters=1,
        #                        kernel_size=self.in_channels,
        #                        padding='same'
        #                        )(reconstruction)

        return logits_seg
        # return logits_seg, reconstruction

    def create_model(self, name='DenseUnet'):
        input_ = tf.keras.layers.Input(shape=self.image_shape, dtype=tf.float32, name="input")
        logits = self.build_network(input_)
        if self.out_channels == 1:
            output_ = tf.keras.activations.sigmoid(logits)
        else:
            output_ = tf.keras.layers.Softmax(name='output')(logits)
        model = tf.keras.models.Model(input_, output_, name=name)
        return model
