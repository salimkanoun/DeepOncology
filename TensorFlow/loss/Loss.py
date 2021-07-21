import tensorflow as tf


#DICE metric
def metric_dice(dim, square = True):
    """
    :param dim: Must be 2 or 3
    https://mediatum.ub.tum.de/doc/1395260/1395260.pdf
    """
    axis = tuple(i for i in range(1, dim+1))

    def dice_similarity_coefficient(y_true, y_pred):
        """
        compute dice score for sigmoid prediction
        :param y_true: true label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)
        :param y_pred: pred label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)
        :return: dice score
        """
        smooth = 0.1
        #y_true = tf.cast(y_true, tf.float32)
        numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=axis)
        if square : 
            #denominator = tf.math.reduce_sum(tf.math.square(y_true) + tf.math.square(y_pred), axis=axis)
            denominator = tf.math.reduce_sum(tf.math.square(y_true), axis=axis) + tf.math.reduce_sum(tf.math.square(y_pred), axis=axis)
        else : 
            #denominator = tf.math.reduce_sum(y_true + y_pred, axis=axis)
            denominateur = tf.math.reduce_sum(y_true, axis=axis) + tf.math.reduce_sum(y_pred, axis=axis)

        return tf.reduce_mean((numerator + smooth) / (denominator + smooth))

    def dice(y_true, y_pred):
        #y_pred = tf.math.round(y_pred)
        y_true = tf.math.round(y_true)
        return dice_similarity_coefficient(y_true, y_pred)

    return dice


#DICE loss
def loss_dice(dim, square=True):
    """[summary]

    Args:
        dim ([type]): [description]
        square (bool, optional): [https://mediatum.ub.tum.de/doc/1395260/1395260.pdf]. Defaults to True.

    Returns:
        [type]: [description]
    """
    axis = tuple(i for i in range(1, dim + 1))

    def dice_similarity_coefficient(y_true, y_pred):
        smooth = 0.1

        y_true = tf.cast(y_true, tf.float32)

        numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=axis)
        if square:
            #denominator = tf.math.reduce_sum(tf.math.square(y_true) + tf.math.square(y_pred), axis=axis)
            denominator = tf.math.reduce_sum(tf.math.square(y_true), axis=axis) + tf.math.reduce_sum(tf.math.square(y_pred), axis=axis)
        else:
            #denominator = tf.math.reduce_sum(y_true + y_pred, axis=axis)
            denominateur = tf.math.reduce_sum(y_true, axis=axis) + tf.math.reduce_sum(y_pred, axis=axis)

        return tf.reduce_mean((numerator + smooth) / (denominator + smooth))

    def loss(y_true, y_pred):
        #y_pred = tf.math.round(y_pred)
        y_true = tf.math.round(y_true)
        return 1.0 - dice_similarity_coefficient(y_true, y_pred)

    return loss


#RCE loss
def rce_loss(y_true, y_pred):
    ce_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    return tf.reduce_mean(ce_loss(y_true, y_pred) + ce_loss(y_pred, y_true))

#TVERSKY Loss 
def tversky_loss(dim, beta):
    """
    https://arxiv.org/abs/1706.05721
    """
    axis = tuple(i for i in range(1, dim + 1))

    def loss(y_true, y_pred):
        smooth = 0.1
        y_true = tf.cast(y_true, tf.float32)

        numerator = tf.reduce_sum(y_true * y_pred, axis=axis)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

        return 1.0 - tf.reduce_sum((numerator + smooth) / (tf.reduce_sum(denominator, axis=axis) + smooth))
    return loss 


#generalized dice loss
def generalized_dice_loss(class_weight, dim):
    axis = tuple(i for i in range(1, dim))

    def generalized_dice(y_true, y_pred):
        smooth = 0.1

        y_true = tf.cast(y_true, tf.float32)

        numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=axis)
        denominator = tf.math.reduce_sum(y_true + y_pred, axis=axis)

        numerator = tf.math.reduce_sum(class_weight * numerator, axis=-1)
        denominator = tf.math.reduce_sum(class_weight * denominator, axis=-1)

        return tf.math.reduce_mean((numerator + smooth) / (denominator + smooth))

    return 1.0 - generalized_dice


def FocalLoss(alpha, gamma):
    """
    https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
    https://arxiv.org/pdf/1708.02002.pdf
    """
    return tf.keras.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma)


def custom_loss_roche(dim):
    """
    https://doi.org/10.1007/s10278-020-00341-1
    """
    axis = tuple(i for i in range(1, dim + 1))

    def roche_loss(y_true, y_pred):

        smooth = 1.0

        y_true = tf.cast(y_true, tf.float32)

        intersection = tf.math.reduce_sum(y_true * y_pred, axis=axis)
        P = tf.math.reduce_sum(y_pred, axis=axis)
        T = tf.math.reduce_sum(y_true, axis=axis)

        dice_coef = (2.0 * intersection + smooth)/(P + T + smooth)
        sensitivity_coef = (intersection + smooth)/(T + smooth)

        mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        mean_abs = mae(y_true, y_pred)

        return tf.reduce_sum(mean_abs - dice_coef - sensitivity_coef)

    return 1.0 - roche_loss


def custom_robust_loss(dim):
    """
    loss = Dice loss + CE + L1
    """
    axis = tuple(i for i in range(1, dim + 1))

    def loss(y_true, y_pred): 

        y_true = tf.cast(y_true, tf.float32)

        # dice
        smooth = 10.0

        numerator = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=axis)
        denominator = tf.math.reduce_sum(tf.math.square(y_true) + tf.math.square(y_pred), axis=(1, 2, 3, 4))
        squared_dice = tf.reduce_mean((numerator + smooth) / (denominator + smooth))

        # CE
        ce_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        # L1
        mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

        return 1.0 - squared_dice + tf.reduce_mean(ce_loss(y_true, y_pred) + mae(y_true, y_pred))

    return loss 



def transform_to_onehot(y_true, y_pred):
    num_classes = y_pred.shape[-1]
    indices = tf.cast(y_true, dtype=tf.int32)
    onehot_labels = tf.one_hot(indices=indices, depth=num_classes, dtype=tf.float32, name='onehot_labels')
    return onehot_labels, y_pred

############ LOSS SURVIVAL 

def get_logLikelihood_LOSS(time, event, time_horizon_dim, y_pred):
    '''
    for uncensored : log of the probabilitie predicted to experience the event at the true time t
    for censored : log of the sum of probabilities predicted to experience the event after the true censoring time t
    '''
    #mask for the log-likelihood loss
    #mask size is [N, time_horizon_dim]
    #    if not censored : one element = 1 (0 elsewhere)
    #    if censored     : fill elements with 1 after the censoring time (for all events)
    #mask = np.zeros([len(time), time_horizon_dim]) 

    mask_uncensored = tf.one_hot(time, time_horizon_dim, dtype=tf.float32)
    mask_censored = tf.one_hot(time+1, time_horizon_dim, dtype= tf.float32)
    
    for i in range(time_horizon_dim):
        mask_censored = mask_censored + tf.one_hot(time+2+i, time_horizon_dim, dtype= tf.float32)
    mask= tf.transpose([event])*mask_uncensored + tf.transpose([1-event])*mask_censored
    #for uncensored: 
    mask=tf.cast(mask, tf.float32) 
    preds= tf.reduce_sum(mask * y_pred, 1)
    tmp1 = event*log(preds)
    #for censored: log \sum P(T>t|x)
    tmp2=(1-event)*log(preds)
    loss=-tf.reduce_sum(tf.reduce_sum(tmp1)+tf.reduce_sum(tmp2))/tf.reduce_sum(tf.where(event==-1., 0., 1.))
    return loss

def get_ranking_LOSS(time, event, time_horizon_dim, y_pred):
    '''
    for pairs acceptables (careful with censored events):
    loss  function η(P(x),P(y)) = exp(−(P(x)−P(y))/σ)
    where P(x) is the sum of probabilities for x to experience the event on time t <= tx -- (true time of x) --
    and P(y) is the sum of probabilities for y to experience the event on time t <= tx
    translated to : a patient who dies at a time s should have a higher risk at time s than a patient who survived longer than s
    '''
    #    mask is required calculate the ranking loss (for pair-wise comparision)
    #    mask size is [N, time_horizon_dim].
    #         1's from start to the event time(inclusive)

    mask = tf.one_hot(time, time_horizon_dim)
    for i in range(time_horizon_dim):
        mask= mask + tf.one_hot(time-i-1, time_horizon_dim)

    sigma = tf.constant(0.1, dtype=tf.float32)
    one_vector = tf.ones_like([time], dtype=tf.float32)
    #event_tf = tf.cast(event, tf.float32)
    time_tf = tf.cast([time],tf.float32)

    mask=tf.cast(tf.transpose(mask),tf.float32)
    I_2 = tf.linalg.diag(event)
    
    #R : r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})
    R = tf.linalg.matmul(y_pred, mask)
    diag_R = tf.linalg.diag_part(R)
    diag_R = tf.reshape(diag_R, [-1, 1]) # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
    R2 = tf.transpose(tf.transpose(diag_R)-R)
    # diag_R-R : R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
    # transpose : R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

    T = tf.nn.relu(tf.sign(tf.matmul(tf.transpose(one_vector), time_tf)-tf.matmul(tf.transpose(time_tf), one_vector)))
    # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

    T2 = tf.matmul(I_2, T)
    loss = tf.reduce_sum(T2 * tf.exp(-R2/sigma))/tf.reduce_sum(T2)
    return loss

def get_loss_survival(time_horizon_dim, alpha, beta, gamma):
    ''' 
    time_horizon_dim : output dimension of the output layer of the model
    loss_survival : returns the loss of the model (log_likelihood loss + ranking loss)
    '''
    def loss_survival(y_true, y_pred):
        y_true = tf.transpose(y_true)
        time = tf.math.abs(y_true[0])
        event = tf.where(y_true[0]> 0, 1,0)
        event = tf.cast(event, tf.float32)
        loss_logLikelihood= get_logLikelihood_LOSS(time, event, time_horizon_dim, y_pred)
        
        loss_ranking=get_ranking_LOSS(time, event, time_horizon_dim, y_pred)
        loss_ranking =  tf.where(tf.math.is_nan(loss_ranking),0.,loss_ranking)
        loss= alpha*loss_logLikelihood + beta*loss_ranking #+gamma*loss_brier

        return loss

    return loss_survival

    def log(x):
        return tf.math.log(x+10**(-4))