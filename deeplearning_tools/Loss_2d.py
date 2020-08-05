import tensorflow as tf


def metric_dice(y_true, y_pred):
    y_pred = tf.math.round(y_pred)
    return dice_similarity_coefficient(y_true, y_pred)


def dice_similarity_coefficient(y_true, y_pred):
    smooth = 1.0
    axis = (1, 2, 3)  # tuple(range(1, len(y_pred.shape)))  # all axis except first

    numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=axis)
    denominator = tf.math.reduce_sum(y_true + y_pred, axis=axis)

    return tf.reduce_mean((numerator + smooth) / (denominator + smooth))


def vnet_dice(y_true, y_pred):
    """
    https://arxiv.org/abs/1606.04797
    """
    smooth = 1.0
    axis = (1, 2, 3)  # tuple(range(1, len(y_pred.shape)))  # all axis except first

    numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=axis)
    denominator = tf.math.reduce_sum(tf.math.square(y_true) + tf.math.square(y_pred), axis=axis)

    return tf.reduce_mean((numerator + smooth) / (denominator + smooth))


def focal_loss(alpha, gamma):
    """
    https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
    https://arxiv.org/pdf/1708.02002.pdf
    """
    return tf.keras.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma)


def densexnet_loss(y_true, y_pred):
    alpha, gamma = 0.9, 3
    return focal_loss(alpha, gamma)(y_true, y_pred) - vnet_dice(y_true, y_pred)

