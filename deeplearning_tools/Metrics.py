import numpy as np
from scipy.spatial.distance import directed_hausdorff, jaccard


def hausdorff_distance(y_true, y_pred):
    """
    hausdorff distance for binary 3D segmentation

    Args :
        Inputs must be ndarray of 0 or 1 (binary)
        :param y_true: true label image of shape (batch_size, z, y, x)
        :param y_pred: pred label image of shape (batch_size, z, y, x)

    :return: hausdorff distance
    """
    true_vol_idx = np.where(y_true)
    pred_vol_idx = np.where(y_pred)

    return max(directed_hausdorff(true_vol_idx, pred_vol_idx)[0], directed_hausdorff(pred_vol_idx, true_vol_idx)[0])


def IoU(y_true, y_pred):
    """
    Intersection over Union aka Jaccard index

    Args :
        Inputs must be ndarray of 0 or 1 (binary)
        :param y_true: true label image of shape (z, y, x)
        :param y_pred: pred label image of shape (z, y, x)

    :return: IoU, float
    """
    return jaccard(y_true.flatten(), y_pred.flatten())


def dice_similarity_coefficient(y_true, y_pred):
    """
    compute dice score for multi-class prediction

    Args :
        :param y_true: true label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)
        :param y_pred: pred label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)

    :return: dice score
    """
    smooth = 0.1

    numerator = 2 * np.sum(y_true * y_pred, axis=(1, 2, 3, 4))
    denominator = np.sum(y_true + y_pred, axis=(1, 2, 3, 4))

    return np.mean((numerator + smooth) / (denominator + smooth))


def sensitivity(y_true, y_pred):
    """
    sensitivity = tp/(tp+fn)

    Args :
        Inputs must be ndarray of 0 or 1 (binary)
        :param y_true: true label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)
        :param y_pred: pred label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)

    :return: sensitivity
    """

    tp = np.sum(y_pred * y_true, axis=(1, 2, 3, 4))
    positive = np.sm(y_true, axis=(1, 2, 3, 4))

    return np.mean(tp / positive)


def AVD(y_true, y_pred, volume_voxel=1.0):
    """
    Average volume difference

    Args :

        :param y_true: true label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)
        :param y_pred: pred label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)
        :param volume_voxel:  volume of one voxel, default to 1.0
    :return: AVD, float
    """
    assert y_true.shape == y_pred.shape

    vol_g = np.sum(y_true, axis=y_true.shape[1:])
    vol_p = np.sum(y_pred, axis=y_pred.shape[1:])

    return volume_voxel * np.mean(abs(vol_g - vol_p) / vol_g)


def apply_threshold(y_pred, threshold=0.5):
    """
    apply threshold to predict mask

    Args :
        :param y_pred: pred segmentation, image of shape (batch_size, z, y, x) or  (batch_size, z, y, x, 1)
        :param threshold: threshold to apply, float
    :return: round mask of shape (batch_size, z, y, x)
    """

    new_y_pred = np.zeros(y_pred.shape, dtype=int)
    new_y_pred[y_pred > threshold] = 1

    return new_y_pred
