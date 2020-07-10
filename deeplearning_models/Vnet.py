
import tensorflow as tf
from .Layers import convolution, down_convolution, up_convolution, get_num_channels


def dropout(x, keep_prob):
    """
    edit in the main
    """
    # tf.keras.layers.Dropout(keep_prob)(x)
    return tf.keras.layers.SpatialDropout3D(keep_prob)(x)


def convolution_block(layer_input, num_convolutions, keep_prob, activation_fn):
    x = layer_input
    n_channels = get_num_channels(x)
    for i in range(num_convolutions):
        x = convolution(x, n_channels, kernel_size=(5, 5, 5))
        # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        x = activation_fn(x)
        # x = tf.keras.layers.Dropout(keep_prob)(x)

    x = x + layer_input
    return x


def convolution_block_2(layer_input, fine_grained_features, num_convolutions, keep_prob, activation_fn):

    x = tf.keras.layers.Concatenate()([layer_input, fine_grained_features])
    n_channels = get_num_channels(layer_input)
    for i in range(num_convolutions):
        x = convolution(x, n_channels, kernel_size=(5, 5, 5))
        # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        x = activation_fn(x)
        # x = tf.keras.layers.Dropout(keep_prob)(x)

    # layer_input = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(layer_input)
    x = x + layer_input
    return x


class VNet(object):
    """
    Implements VNet architecture https://arxiv.org/abs/1606.04797
    """
    def __init__(self,
                 image_shape,
                 num_classes,
                 keep_prob=1.0,
                 kernel_size=(5, 5, 5),
                 num_channels=16,
                 num_levels=4,
                 num_convolutions=(1, 2, 3, 3),
                 bottom_convolutions=3,
                 activation=tf.keras.layers.PReLU()):
        """
        :param image_shape: Shape of the input image
        :param num_classes: Number of output classes.
        :param kernel_size: Size of the convolutional patch
        :param keep_prob: Dropout keep probability, set to 1.0 if not training or if no dropout is desired.
        :param num_channels: The number of output channels in the first level, this will be doubled every level.
        :param num_levels: The number of levels in the network. Default is 4 as in the paper.
        :param num_convolutions: An array with the number of convolutions at each level.
        :param bottom_convolutions: The number of convolutions at the bottom level of the network.
        :param activation_fn: The activation function.
        """
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        assert num_levels == len(num_convolutions)
        self.num_levels = num_levels
        self.num_convolutions = num_convolutions
        self.bottom_convolutions = bottom_convolutions
        self.activation_fn = activation

    def build_network(self, input_):
        x = input_
        keep_prob = self.keep_prob

        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input
        # channel
        input_channels = int(x.get_shape()[-1])
        if input_channels == 1:
            x = tf.keras.backend.tile(x, [1, 1, 1, 1, self.num_channels])
        else:
            x = convolution(x, self.num_channels, kernel_size=self.kernel_size)
            # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
            x = self.activation_fn(x)

        forwards = list()
        for l in range(self.num_levels):
            x = convolution_block(x, self.num_convolutions[l], keep_prob, activation_fn=self.activation_fn)
            forwards.append(x)
            x = down_convolution(x, factor=2, kernel_size=(2, 2, 2))
            # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
            x = self.activation_fn(x)

        x = convolution_block(x, self.bottom_convolutions, keep_prob, activation_fn=self.activation_fn)

        for l in reversed(range(self.num_levels)):
            f = forwards[l]
            x = up_convolution(x, factor=2, kernel_size=(2, 2, 2))
            # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
            x = self.activation_fn(x)

            x = convolution_block_2(x, f, self.num_convolutions[l], keep_prob, activation_fn=self.activation_fn)

        # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        logits = convolution(x, self.num_classes)

        return logits

    def create_model(self):
        input_ = tf.keras.layers.Input(shape=self.image_shape, dtype=tf.float32, name="input")
        logits = self.build_network(input_)
        output_ = tf.keras.layers.Softmax(name='output')(logits)
        model = tf.keras.models.Model(input_, output_, name='VNet')
        return model

