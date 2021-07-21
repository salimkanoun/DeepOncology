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