import tensorflow as tf

import numpy as np
import SimpleITK as sitk

from scipy.stats import truncnorm
import random

from .preprocessing import PreprocessorPETCT
from .data_augmentation import DataAugmentor


class DataGenerator(tf.keras.utils.Sequence):
    """
    Read, preprocess and flow the PET scan, CT scan and mask

    """

    def __init__(self,
                 images_paths,
                 labels_path,
                 batch_size=1,
                 shuffle=True,
                 augmentation=False,
                 target_shape=None,
                 target_voxel_spacing=None,
                 resize=True,
                 normalize=True,
                 origin='head'):
        """
        :param images_paths:         list of tuple : [(PET_id, CT_id), ...]
        :param labels_path:          list, [MASK_id, ...]
        :param batch_size:           int
        :param shuffle:              bool
        :param augmentation:         bool
        :param target_shape:         tuple, shape of generated PET, CT or MASK scan: (z, y, x) (368, 128, 128)
        :param target_voxel_spacing: tuple, resolution of the generated PET, CT or MASK scan : (4.8, 4.8, 4.8)
        :param resize:               bool
        :param normalize:            bool
        """
        self.images_paths = images_paths
        self.number_channels = 2  # PET + CT scan

        self.labels_path = labels_path
        self.labels_names = ['Background', 'Lymphoma']
        self.labels_numbers = [0, 1]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augment = augmentation

        self.dtypes = {'pet_img': sitk.sitkFloat32,
                       'ct_img': sitk.sitkFloat32,
                       'mask_img': sitk.sitkUInt8}
        self.default_threshold = 'auto'
        self.preprocessor = PreprocessorPETCT(target_shape=target_shape,
                                              target_voxel_spacing=target_voxel_spacing,
                                              resize=resize,
                                              normalize=normalize,
                                              origin=origin)
        self.data_augmentor = DataAugmentor()

    def __len__(self):
        """
        :return: int, the number of batches per epoch
        """
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.images_paths))
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
        Y_batch = []
        for idx in indexes:
            # select data
            PET_id, CT_id = self.images_paths[idx]
            MASK_id = self.labels_path[idx]

            # load, normalize and resample images
            PET_img, CT_img, MASK_img = self.preprocess_data(PET_id, CT_id, MASK_id)

            if self.augment:
                PET_img, CT_img, MASK_img = self.augment_data(PET_img, CT_img, MASK_img)

            # convert to numpy array
            PET_array = sitk.GetArrayFromImage(PET_img)
            CT_array = sitk.GetArrayFromImage(CT_img)
            MASK_array = sitk.GetArrayFromImage(MASK_img)

            # concatenate PET and CT
            PET_CT_array = np.stack((PET_array, CT_array), axis=-1)

            # add 1 channel to mask
            MASK_array = np.expand_dims(MASK_array, axis=-1)

            # add it to the batch
            X_batch.append(PET_CT_array)
            Y_batch.append(MASK_array)

        return np.array(X_batch), np.array(Y_batch)

    def preprocess_data(self, pet_id, ct_id, mask_id):
        """
        Args :
            :param pet_id: string, path to PET scan
            :param ct_id: string, path to CT scan
            :param mask_id: string, path to MASK

        :return: return preprocessed PET, CT, MASK img
        """

        pet_img = self.read_PET(pet_id)
        ct_img = self.read_CT(ct_id)
        mask_img = self.read_MASK(mask_id)

        if self.augment:
            threshold = self.generate_random_threshold()
        else:
            threshold = self.default_threshold

        output = self.preprocessor({'pet_img': pet_img, 'ct_img': ct_img, 'mask_img': mask_img},
                                   threshold=threshold)
        return output['pet_img'], output['ct_img'], output['mask_img']

    def augment_data(self, pet_img, ct_img, mask_img):
        output = self.data_augmentor({'pet_img': pet_img, 'ct_img': ct_img, 'mask_img': mask_img})
        return output['pet_img'], output['ct_img'], output['mask_img']

    def read_PET(self, filename):
        return sitk.ReadImage(filename, self.dtypes['pet_img'])

    def read_CT(self, filename):
        return sitk.ReadImage(filename, self.dtypes['ct_img'])

    def read_MASK(self, filename):
        return sitk.ReadImage(filename, self.dtypes['mask_img'])

    def save_img(self, img, filename):
        """
        :param img: image, simple itk image
        :param filename: path/to/file.nii, where to save the image
        """
        sitk.WriteImage(img, filename)

    @staticmethod
    def generate_random_bool(p):
        """
        :param p : float between 0-1, probability
        :return: True if a probobility of p
        """
        return random.random() < p

    def generate_random_threshold(self):
        if self.generate_random_bool(0.5):
            lower, upper = 2.5, 4.0
            mu, std = 3.0, 0.5

            a, b = (lower - mu) / std, (upper - mu) / std
            return truncnorm.rvs(a, b, loc=mu, scale=std)
        else:
            return 'auto'
