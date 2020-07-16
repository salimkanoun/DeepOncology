import numpy as np
from scipy.spatial.distance import directed_hausdorff


def hausdorff_distance(y_true, y_pred):
    """
    hausdorff distance for binary 3D segmentation

    Args :
        Inputs must be ndarray of 0 or 1
        :param y_true: true label image of shape (batch_size, z, y, x)
        :param y_pred: pred label image of shape (batch_size, z, y, x)

    :return: hausdorff distance
    """
    true_vol_idx = np.where(y_true)
    pred_vol_idx = np.where(y_pred)

    return max(directed_hausdorff(true_vol_idx , pred_vol_idx)[0], directed_hausdorff(pred_vol_idx, true_vol_idx)[0])


def dice_similarity_coefficient(y_true, y_pred):
    """
    compute dice score for multi-class prediction

    Args :
        :param y_true: true label image of shape (batch_size, z, y, x, num_class)
        :param y_pred: pred label image of shape (batch_size, z, y, x, num_class)

    :return: dice score
    """
    smooth = 0.1

    numerator = 2 * np.sum(y_true * y_pred, axis=(1, 2, 3, 4))
    denominator = np.sum(y_true + y_pred, axis=(1, 2, 3, 4))

    return np.sum((numerator + smooth) / (denominator + smooth))
