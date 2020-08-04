import tensorflow as tf


# def sensitivity(y_true, y_pred):
#     """
#     sensitivity = tp/(tp+fn)
#     """
#     y_pred = tf.math.round(y_pred)
#
#     tp = tf.keras.backend.sum(y_pred * y_true, axis=(1, 2, 3))
#     denominator = tf.keras.backend.sum(y_true, axis=(1, 2, 3))
#     return tf.keras.backend.mean(tp/denominator)
#
#
# def specificity(y_true, y_pred):
#     """
#     specifity = tn/(tn+fp)
#     """
#     y_pred = tf.math.round(y_pred)
#     pass


# def ppv(y_true, y_pred):
#     """
#     Positive Predictive value
#     ppv = tp /(tp +fp)
#     """
#     smooth = 1.0
#
#     y_pred = tf.math.round(y_pred)
#
#     tp = tf.math.reduce_sum(y_true * y_pred, axis=(1, 2, 3, 4))
#     P = tf.math.reduce_sum(y_pred, axis=(1, 2, 3, 4))
#
#     return tf.math.reduce_mean((tp + smooth)/(P + smooth))

def metric_precision(y_true, y_pred):
    y_pred = tf.math.round(y_pred)
    tp = tf.keras.backend.sum(y_pred * y_true, axis=(1, 2, 3))


def metric_dice(y_true, y_pred):
    y_pred = tf.math.round(y_pred)
    return dice_similarity_coefficient(y_true, y_pred)


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

    :param y_true: true label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)
    :param y_pred: pred label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)

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

        y_true = tf.cast(y_true, dtype=tf.float32)

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
        y_true = tf.cast(y_true, dtype=tf.float32)

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
    y_true = tf.cast(y_true, dtype=tf.float32)
    return 1.0 - binary_dice_similarity_coefficient(y_true, y_pred)


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    return 1.0 - dice_similarity_coefficient(y_true, y_pred)


def custom_loss3D_roche(y_true, y_pred):
    """
    https://doi.org/10.1007/s10278-020-00341-1
    """
    smooth = 1.0

    y_true = tf.cast(y_true, dtype=tf.float32)

    intersection = tf.math.reduce_sum(y_true * y_pred, axis=(1, 2, 3, 4))
    P = tf.math.reduce_sum(y_pred, axis=(1, 2, 3, 4))
    T = tf.math.reduce_sum(y_true, axis=(1, 2, 3, 4))

    dice_coef = (2.0 * intersection + smooth)/( P + T + smooth)
    sensitivity_coef = (intersection + smooth)/(T + smooth)
    mean_abs = tf.reduce_sum(tf.math.abs(y_true, y_pred), axis=(1, 2, 3, 4))

    return tf.reduce_sum(mean_abs - dice_coef - sensitivity_coef)


def custom_loss_DenseX(y_true, y_pred):
    """
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8946601
    """
    alpha, gamma = 0.9, 3

    y_true = tf.cast(y_true, dtype=tf.float32)

    loss_f = FocalLoss(alpha, gamma)
    loss_d = vnet_dice_loss(y_true, y_pred)

    return loss_d + loss_f(y_true, y_pred)


def vnet_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
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



