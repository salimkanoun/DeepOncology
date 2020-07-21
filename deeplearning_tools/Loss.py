import tensorflow as tf


def binary_dice_similarity_coefficient(y_true, y_pred):
    """
    Compute dice score

    Args :
        :param y_true: true label image of shape (batch_size, z, y, x)
        :param y_pred: pred label image of shape (batch_size, z, y, x)

    :return: dice score
    """
    smooth = 0.1

    numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.math.reduce_sum(y_true + y_pred, axis=(1, 2, 3))

    return tf.math.reduce_mean((numerator + smooth) / (denominator + smooth))


def dice_similarity_coefficient(y_true, y_pred):
    """
    compute dice score for multi-class prediction

    :param y_true: true label image of shape (batch_size, z, y, x, num_class)
    :param y_pred: pred label image of shape (batch_size, z, y, x, num_class)

    :return: dice score
    """
    smooth = 0.1

    numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=(1, 2, 3, 4))
    denominator = tf.math.reduce_sum(y_true + y_pred, axis=(1, 2, 3, 4))

    return tf.reduce_mean((numerator + smooth) / (denominator + smooth))


def vnet_dice(y_true, y_pred):
    """
    https://arxiv.org/abs/1606.04797
    """
    smooth = 0.1

    numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=(1, 2, 3, 4))
    denominator = tf.math.reduce_sum(tf.math.square(y_true) + tf.math.square(y_pred), axis=(1, 2, 3, 4))

    return tf.reduce_mean((numerator + smooth) / (denominator + smooth))


def generalized_dice_loss(class_weight):
    def generalized_dice(y_true, y_pred):
        smooth = 0.1

        numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.math.reduce_sum(y_true + y_pred, axis=(1, 2, 3))

        numerator = tf.math.reduce_sum(class_weight * numerator, axis=-1)
        denominator = tf.math.reduce_sum(class_weight * denominator, axis=-1)

        return tf.math.reduce_mean((numerator + smooth) / (denominator + smooth))

    return 1.0 - generalized_dice


def tversky_loss(beta):
    """
    https://arxiv.org/abs/1706.05721
    """
    def loss(y_true, y_pred):
        smooth = 0.1
        numerator = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3, 4))
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

        return 1.0 - tf.reduce_sum((numerator + smooth) / (tf.reduce_sum(denominator, axis=(1, 2, 3, 4)) + smooth))

    return loss


def FocalLoss(alpha, gamma):
    """
    https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
    https://arxiv.org/pdf/1708.02002.pdf
    """
    return tf.keras.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma)


def binary_dice_loss(y_true, y_pred):
    return 1.0 - binary_dice_similarity_coefficient(y_true, y_pred)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_similarity_coefficient(y_true, y_pred)


def vnet_dice_loss(y_true, y_pred):
    return 1.0 - vnet_dice(y_true, y_pred)


def transform_to_onehot(y_true, y_pred):
    num_classes = y_pred.shape[-1]

    indices = tf.cast(y_true, dtype=tf.int32)
    onehot_labels = tf.one_hot(indices=indices, depth=num_classes, dtype=tf.float32, name='onehot_labels')
    return onehot_labels, y_pred


class CustomLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        # return dice_loss(y_true, y_pred)
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        return dice_loss(y_true, y_pred) + tf.keras.losses.BinaryCrossentropy(y_true, y_pred)



