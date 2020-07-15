import tensorflow as tf


def binary_dice_similarity_coefficient(y_true, y_pred):
    """
    Compute dice score

    :param y_true: true label image of shape (batch_size, z, y, x)
    :param y_pred: pred label image of shape (batch_size, z, y, x)

    :return: dice score
    """
    smooth = 0.1

    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))

    return tf.reduce_sum((numerator + smooth) / (denominator + smooth))


def dice_similarity_coefficient(y_true, y_pred):
    """
    compute dice score for multi-class prediction

    :param y_true: true label image of shape (batch_size, z, y, x, num_class)
    :param y_pred: pred label image of shape (batch_size, z, y, x, num_class)

    :return: dice score
    """
    smooth = 0.1

    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3, 4))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3, 4))

    return tf.reduce_sum((numerator + smooth) / (denominator + smooth))


def generalized_dice_loss(class_weight):
    def generalized_dice(y_true, y_pred):
        smooth = 0.1

        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))

        numerator = tf.reduce_sum(class_weight * numerator, axis=-1)
        denominator = tf.reduce_sum(class_weight * denominator, axis=-1)

        return tf.reduce_sum((numerator + smooth) / (denominator + smooth))

    return 1.0 - generalized_dice


def binary_dice_loss(y_true, y_pred):
    return 1.0 - binary_dice_similarity_coefficient(y_true, y_pred)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_similarity_coefficient(y_true, y_pred)


class CustomLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        num_classes = y_pred.shape[-1]

        indices = tf.cast(y_true, dtype=tf.int32)
        onehot_labels = tf.one_hot(indices=indices, depth=num_classes, dtype=tf.float32, name='onehot_labels')

        return binary_dice_loss(onehot_labels, y_pred) + tf.keras.losses.CategoricalCrossentropy(onehot_labels, y_pred)
        # return binary_dice_loss(onehot_labels, y_pred) + tf.keras.losses.CategoricalCrossentropy(onehot_labels, y_pred)



