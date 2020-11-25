
import tensorflow as tf
from .Layers import convolution, down_convolution, up_convolution, get_num_channels


def dropout(x, keep_prob):
    # tf.keras.layers.Dropout(1.0 - keep_prob)(x)
    return tf.keras.layers.SpatialDropout3D(1.0 - keep_prob)(x)


def convolution_block(layer_input, num_convolutions, kernel_size, keep_prob, activation_fn):
    x = layer_input
    n_channels = get_num_channels(x)
    for i in range(num_convolutions):
        x = convolution(x, n_channels, kernel_size=kernel_size)
        # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        x = activation_fn(x)
        if keep_prob < 1.0:
            x = dropout(x, keep_prob)

    x = x + layer_input
    return x


def convolution_block_2(layer_input, fine_grained_features, num_convolutions, kernel_size, keep_prob, activation_fn):

    x = tf.keras.layers.Concatenate()([layer_input, fine_grained_features])
    n_channels = get_num_channels(layer_input)
    for i in range(num_convolutions):
        x = convolution(x, n_channels, kernel_size=kernel_size)
        # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        x = activation_fn(x)
        if keep_prob < 1.0:
            x = dropout(x, keep_prob)

    # layer_input = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(layer_input)
    x = x + layer_input
    return x


class VNet(object):
    """
    Implements VNet architecture https://arxiv.org/abs/1606.04797
    """
    def __init__(self,
                 image_shape,
                 in_channels,
                 out_channels,
                 channels_last=True,
                 keep_prob=1.0,
                 keep_prob_last_layer=1.0,
                 kernel_size=(5, 5, 5),
                 num_channels=16,
                 num_levels=4,
                 num_convolutions=(1, 2, 3, 3),
                 bottom_convolutions=3,
                 activation=tf.keras.layers.PReLU(),
                 activation_last_layer='sigmoid'):
        """
        :param image_shape: Shape of the input image
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param channels_last: bool, set to True for channels last format
        :param kernel_size: Size of the convolutional patch
        :param keep_prob: Dropout keep probability in the conv layer,
                            set to 1.0 if not training or if no dropout is desired.
        :param keep_prob_last_layer: Dropout keep probability in the last conv layer,
                                    set to 1.0 if not training or if no dropout is desired.
        :param num_channels: The number of output channels in the first level, this will be doubled every level.
        :param num_levels: The number of levels in the network. Default is 4 as in the paper.
        :param num_convolutions: An array with the number of convolutions at each level.
        :param bottom_convolutions: The number of convolutions at the bottom level of the network.
        :param activation: The activation function.
        :param activation_last_layer: The activation function used in the last layer of the cnn.
                                      Set to None to return logits.
        """
        self.image_shape = image_shape
        assert len(image_shape) == 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels_last = channels_last
        assert channels_last  # channels_last=False is not supported
        self.kernel_size = kernel_size
        self.keep_prob = keep_prob
        self.keep_prob_last_layer = keep_prob_last_layer
        self.num_channels = num_channels
        assert num_levels == len(num_convolutions)
        self.num_levels = num_levels
        self.num_convolutions = num_convolutions
        self.bottom_convolutions = bottom_convolutions
        self.activation_fn = activation
        if isinstance(self.activation_fn, str):
            self.activation_fn = self.activation_fn.lower()
            self.activation_fn = tf.keras.layers.PReLU() if self.activation_fn == 'prelu' \
                else tf.keras.activations.get(self.activation_fn)
        self.activation_last_layer = activation_last_layer.lower() if isinstance(activation_last_layer, str) \
            else activation_last_layer

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
            x = convolution_block(x, self.num_convolutions[l], self.kernel_size, keep_prob, activation_fn=self.activation_fn)
            forwards.append(x)
            x = down_convolution(x, factor=2, kernel_size=(2, 2, 2))
            # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
            x = self.activation_fn(x)

        x = convolution_block(x, self.bottom_convolutions, self.kernel_size, keep_prob, activation_fn=self.activation_fn)

        for l in reversed(range(self.num_levels)):
            f = forwards[l]
            x = up_convolution(x, factor=2, kernel_size=(2, 2, 2))
            # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
            x = self.activation_fn(x)

            x = convolution_block_2(x, f, self.num_convolutions[l], self.kernel_size, keep_prob, activation_fn=self.activation_fn)

        # x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        if self.keep_prob_last_layer < 1.0:
            x = tf.keras.layers.Dropout(1.0 - self.keep_prob_last_layer)(x)
        logits = convolution(x, self.out_channels)

        return logits

    def create_model(self):
        input_shape = tuple(list(self.image_shape) + [self.in_channels])
        input_ = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32, name="input")
        logits = self.build_network(input_)
        if self.activation_last_layer is None:
            output_ = logits
        elif self.activation_last_layer == 'sigmoid':
            output_ = tf.keras.activations.sigmoid(logits)
        elif self.activation_last_layer == 'softmax':
            output_ = tf.keras.layers.Softmax(name='output')(logits)
        else:
            output_ = self.activation_last_layer(logits)
        model = tf.keras.models.Model(input_, output_, name='VNet')
        return model

