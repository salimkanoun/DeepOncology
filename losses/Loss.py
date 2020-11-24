import tensorflow as tf


def metrice_dice(dim):
    """
    :param dim: Must be 2 or 3
    """
    axis = tuple(i for i in range(1, dim+1))

    def dice_similarity_coefficient(y_true, y_pred):
        """
        compute dice score for sigmoid prediction

        :param y_true: true label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)
        :param y_pred: pred label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)

        :return: dice score
        """
        smooth = 1.0

        numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=axis)
        denominator = tf.math.reduce_sum(y_true + y_pred, axis=axis)

        return tf.reduce_mean((numerator + smooth) / (denominator + smooth))

    def dice(y_true, y_pred):
        y_pred = tf.math.round(y_pred)
        y_true = tf.math.round(y_true)
        return dice_similarity_coefficient(y_true, y_pred)

    return dice


def loss_dice(dim, squared=True):
    axis = tuple(i for i in range(1, dim + 1))

    def dice(y_true, y_pred):
        smooth = 1.0

        y_true = tf.cast(y_true, dtype=tf.float32)

        numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=axis)
        if squared:
            denominator = tf.math.reduce_sum(y_true + y_pred, axis=axis)
        else:
            denominator = tf.math.reduce_sum(tf.math.square(y_true) + tf.math.square(y_pred), axis=axis)

        return tf.reduce_mean((numerator + smooth) / (denominator + smooth))

    return dice


def rce_loss(y_true, y_pred):
    ce_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    return tf.reduce_mean(ce_loss(y_true, y_pred) + ce_loss(y_pred, y_true))


def tversky_loss(dim, beta):
    """
    https://arxiv.org/abs/1706.05721
    """
    axis = tuple(i for i in range(1, dim + 1))

    def loss(y_true, y_pred):
        smooth = 0.1
        y_true = tf.cast(y_true, dtype=tf.float32)

        numerator = tf.reduce_sum(y_true * y_pred, axis=axis)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

        return 1.0 - tf.reduce_sum((numerator + smooth) / (tf.reduce_sum(denominator, axis=axis) + smooth))

    return loss




