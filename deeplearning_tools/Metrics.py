import numpy as np
from scipy.spatial.distance import directed_hausdorff


def hausdorff_distance(y_true, y_pred):
    """
    hausdorff distance for binary 3D segmentation

    :param y_true: true label image of shape (batch_size, z, y, x)
    :param y_pred: pred label image of shape (batch_size, z, y, x)

    """
    sum = 0
    n_examples = y_true.shape[0]
    for n in range(n_examples):
        sum += directed_hausdorff(y_true[0], y_pred[0])[0]

    return sum/n_examples


def dice_similarity_coefficient(y_true, y_pred):
    """
    compute dice score for multi-class prediction

    :param y_true: true label image of shape (batch_size, z, y, x, num_class)
    :param y_pred: pred label image of shape (batch_size, z, y, x, num_class)

    :return: dice score
    """
    smooth = 0.1

    numerator = 2 * np.sum(y_true * y_pred, axis=(1, 2, 3, 4))
    denominator = np.sum(y_true + y_pred, axis=(1, 2, 3, 4))

    return np.sum((numerator + smooth) / (denominator + smooth))
