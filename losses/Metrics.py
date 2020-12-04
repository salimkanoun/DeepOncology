import numpy as np
from scipy.spatial.distance import directed_hausdorff, jaccard

#metrics with ndarray 


def hausdorff_distance(y_true, y_pred):
    """
    hausdorff distance for binary 3D segmentation

    Args :
        Inputs must be ndarray of 0 or 1 (binary)
        :param y_true: true label image of shape (spatial_dim1, spatial_dim2, ..., spatial_dim3)
        :param y_pred: pred label image of shape (spatial_dim1, spatial_dim2, ..., spatial_dim3)

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


def metric_dice(y_true, y_pred, axis=(1, 2, 3, 4)):
    """
    compute dice score for multi-class prediction

    Args :
        :param y_true: true label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)
        :param y_pred: pred label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)
        :param axis: tuple,

    :return: dice score, ndarray of shape (batch_size,)
    """
    smooth = 0.1

    y_true = np.round(y_true)
    y_pred = np.round(y_pred)

    numerator = 2 * np.sum(y_true * y_pred, axis=axis)
    denominator = np.sum(y_true + y_pred, axis=axis)

    return (numerator + smooth) / (denominator + smooth)


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
    positive = np.sum(y_true, axis=(1, 2, 3, 4))

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

