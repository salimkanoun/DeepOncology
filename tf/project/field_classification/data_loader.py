import tensorflow as tf
import numpy as np
import SimpleITK as sitk 
from tools.field_classification.instance import *

class DataGeneratorFromDict(tf.keras.utils.Sequence):
    """data generator for model.fit()

    Args:
        tf ([type]): [description]
    """

    def __init__(self, images_paths, batch_size = 1, shuffle=True, x_keys = 'ct_img', y_keys = ['upper_limit', 'lower_limit', 'right_arm', 'left_arm']):
        self.images_paths = images_paths 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.x_keys = x_keys 
        self.y_keys = y_keys 


    def __len__(self):
        """
        :return: int, the number of batches per epoch
        """
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.array(list(self.images_paths.keys())) if isinstance(self.images_paths, dict) else np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: int, position of the batch in the Sequence
        :return: tuple of numpy array, (X_batch, Y_batch) of shape (batch_size, ...)
        """

        # select indices of data for next batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        # prepare the batch
        X_batch = []
        Y1_batch = []
        Y2_batch = []
        Y3_batch = []
        Y4_batch = []
        for idx in indexes:
            img_dict = self.images_paths[idx]
            X_batch.append(sitk.GetArrayFromImage(sitk.ReadImage(img_dict[self.x_keys])))
            encoded_instance = encoding_instance(img_dict)
            Y1_batch.append([encoded_instance[0]])
            Y2_batch.append([encoded_instance[1]])
            Y3_batch.append([encoded_instance[2]])
            Y4_batch.append([encoded_instance[3]])

        X_batch = np.array(X_batch)
        Y1_batch = np.array(Y1_batch)
        Y2_batch = np.array(Y2_batch)
        Y3_batch = np.array(Y3_batch)
        Y4_batch = np.array(Y4_batch)
        return X_batch, {'head':Y1_batch, 'legs':Y2_batch, 'right_arm':Y3_batch, 'left_arm':Y4_batch}
    

    