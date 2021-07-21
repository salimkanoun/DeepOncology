import numpy as np
from scipy.spatial.distance import directed_hausdorff, jaccard
import tensorflow as tf
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
    #y_true = np.round(y_true)
    #y_pred = np.round(y_pred)
    numerator = 2 * np.sum(y_true * y_pred, axis=axis)

    denominator = np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis)
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




def metric_cindex(time_horizon_dim, tied_tol=1e-2):
    
    def cindex(y_true, y_pred):
        y_true = tf.transpose(y_true)

        time = tf.math.abs(y_true[0])
        event = tf.where(time > 0, 1,0)
        event = tf.cast(event, tf.float32)
        '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
        '''
        mask = tf.one_hot(tf.cast(time, tf.int32), time_horizon_dim)
        for i in range(time_horizon_dim):
            mask= tf.cast(mask, tf.int32) + tf.cast(tf.one_hot(tf.cast(time, tf.int32)-1-i, time_horizon_dim), tf.int32)
        mask=tf.cast(mask, tf.float32)
        mat_A =tf.tile([time], [tf.squeeze(tf.shape(time)),1])
        mat_A = tf.transpose([time])-mat_A
        mat_A = tf.where(mat_A<0, 1, 0)

        t_pred = tf.linalg.matmul(y_pred, tf.transpose(mask))
        pred_diag = tf.linalg.diag_part(t_pred)
        mat_Q = tf.transpose(t_pred-tf.transpose(pred_diag))
        mat_Q= tf.where(mat_Q<0, 1, 0)
        
        mat_N= tf.ones_like(tf.linalg.diag(time))
        mat_N = tf.transpose([event])*mat_N

        mat_A=tf.cast(mat_A,tf.float64)
        mat_Q=tf.cast(mat_Q,tf.float64)
        mat_N=tf.cast(mat_N,tf.float64)
        #########################################
        mat_time= tf.tile([time], [tf.squeeze(tf.shape(time)),1])
        mat_time= tf.transpose([time])-mat_time
        time_equal= tf.cast(tf.where(mat_time==0, 1, 0), tf.float32)
        
        time_equal= tf.transpose(event*tf.transpose(time_equal))*event
        diag_time_equal= tf.linalg.diag(tf.linalg.diag_part(time_equal))
        time_equal= time_equal-diag_time_equal
        num_tied_tot=tf.reduce_sum(time_equal)/2

        t_pred= tf.linalg.matmul(y_pred, tf.transpose(mask))
        diag_pred= tf.linalg.diag_part(t_pred)
        preds_equal= abs(t_pred-diag_pred)
        preds_equal= tf.where(preds_equal <= tied_tol, 0.5, 0)
        ties = preds_equal * tf.cast(time_equal, tf.float32)
        num_tied_true = tf.reduce_sum(ties)/2
        
        Num= tf.reduce_sum((mat_A*mat_N)*mat_Q)+(int(num_tied_true)/2)
        Den= tf.reduce_sum(mat_A*mat_N)+tf.cast(num_tied_tot, tf.float64)

        if Num != 0.0 and Den != 0.0:
            resultat=float(Num/Den)
        else: 
            resultat = 0.
        return resultat
    return cindex