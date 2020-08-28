import tensorflow as tf

import numpy as np
import SimpleITK as sitk
from skimage import filters

import random
from scipy.stats import truncnorm

from .data_augmentation import DataAugmentor
from math import pi


class InputPipeline(object):
    """
    preprocess the PET scan, CT scan and mask
    """

    def __init__(self,
                 target_shape=None,
                 target_voxel_spacing=None,
                 normalize=True,
                 augment=None
                 ):
        self.target_shape = target_shape[::-1]  # [z, y, x] to [x, y, z]
        self.target_voxel_spacing = target_voxel_spacing[::-1]  # [z, y, x] to [x, y, z]
        self.target_direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        self.default_value = {'pet_img': 0.0,  # before normalize
                              'ct_img': -1000.0,
                              'mask_img': 0}

        self.dtypes = {'pet_img': sitk.sitkFloat32,
                       'ct_img': sitk.sitkFloat32,
                       'mask_img': sitk.sitkUInt8}

        self.interpolator = {'pet_img': sitk.sitkBSpline,
                             'ct_img': sitk.sitkBSpline,
                             'mask_img': sitk.sitkNearestNeighbor}

        self.normalize = normalize
        self.augment = augment

    def __call__(self, inputs, threshold=None):
        return self.transform(inputs, threshold)

    def transform(self, images, threshold=None):

        if threshold is not None and len(images) == 3:
            # get mask from ROI
            images['mask_img'] = self.roi2mask(images['mask_img'], images['pet_img'], threshold=threshold)

        # resample to same shape and spacing resolution
        images = self.resample_images(images)

        if self.augment is not None:
            images = self.augment(images)

        if self.normalize:
            images['pet_img'] = self.normalize_PET(images['pet_img'])
            images['ct_img'] = self.normalize_CT(images['ct_img'])

        return images

    @staticmethod
    def normalize_PET(pet_img):
        intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
        intensityWindowingFilter.SetOutputMaximum(1)
        intensityWindowingFilter.SetOutputMinimum(0)
        windowMax = 25
        windowMin = 0
        intensityWindowingFilter.SetWindowMaximum(windowMax)
        intensityWindowingFilter.SetWindowMinimum(windowMin)
        return intensityWindowingFilter.Execute(pet_img)
        # return sitk.ShiftScale(pet_img, shift=0.0, scale=1. / 25.0)

    @staticmethod
    def normalize_CT(ct_img):
        intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
        intensityWindowingFilter.SetOutputMaximum(1)
        intensityWindowingFilter.SetOutputMinimum(0)
        windowMax = 1024
        windowMin = -1024
        intensityWindowingFilter.SetWindowMaximum(windowMax)
        intensityWindowingFilter.SetWindowMinimum(windowMin)
        return intensityWindowingFilter.Execute(ct_img)

    def resample_img(self, img, new_origin, target_shape, default_value, interpolator):
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(self.target_direction)
        transformation.SetOutputOrigin(new_origin)
        transformation.SetOutputSpacing(self.target_voxel_spacing)
        transformation.SetSize(target_shape)

        transformation.SetDefaultPixelValue(default_value)
        transformation.SetInterpolator(interpolator)

        return transformation.Execute(img)

    def get_max_height(self, pet_size, pet_spacing):
        # pet_spacing = pet_img.GetSpacing()
        # pet_origin = pet_img.GetOrigin()

        return min(pet_size[2] * pet_spacing[2], self.target_shape[2] * self.target_voxel_spacing[2])

    def compute_new_origin_head2hip(self, pet_img):
        new_shape = self.target_shape
        new_spacing = self.target_voxel_spacing
        pet_size = pet_img.GetSize()
        pet_spacing = pet_img.GetSpacing()
        pet_origin = pet_img.GetOrigin()
        height = self.get_max_height(pet_size, pet_spacing)
        new_origin = (pet_origin[0] + 0.5 * pet_size[0] * pet_spacing[0] - 0.5 * new_shape[0] * new_spacing[0],
                      pet_origin[1] + 0.5 * pet_size[1] * pet_spacing[1] - 0.5 * new_shape[1] * new_spacing[1],
                      pet_origin[2] + 1.0 * pet_size[2] * pet_spacing[2] - 1.0 * height)
        return new_origin

    def resample_images(self, inputs_img):
        """
        resample and reshape PET, CT and MASK to the same origin, direction, spacing and shape
        """
        # compute transformation parameters
        new_origin = self.compute_new_origin_head2hip(inputs_img['pet_img'])
        height = self.get_max_height(inputs_img['pet_img'].GetSize(), inputs_img['pet_img'].GetSpacing())
        target_shape = (self.target_shape[0], self.target_shape[1], int(height / self.target_voxel_spacing[2]))

        # apply transformation : resample and reshape
        resampled_img = dict()
        for key, img in inputs_img.items():
            resampled_img[key] = self.resample_img(img, new_origin, target_shape,
                                                   self.default_value[key], self.interpolator[key])

        return resampled_img

    @staticmethod
    def roi2mask(mask_img, pet_img, threshold='auto'):
        """
        Generate the mask from the ROI of the pet scan
        Args:
            :param mask_img: sitk image, raw mask (i.e ROI)
            :param pet_img: sitk image, the corresponding pet scan
            :param threshold: threshold to apply to the ROI to get the tumor segmentation.
                    if set to 'auto', it will take 42% of the maximum
        :return: sitk image, the ground truth segmentation
        """
        # transform to numpy
        mask_array = sitk.GetArrayFromImage(mask_img)
        pet_array = sitk.GetArrayFromImage(pet_img)

        # get 3D meta information
        if len(mask_array.shape) == 3:
            mask_array = np.expand_dims(mask_array, axis=0)

            origin = mask_img.GetOrigin()
            spacing = mask_img.GetSpacing()
            direction = tuple(mask_img.GetDirection())
            size = mask_img.GetSize()
        else:
            origin = mask_img.GetOrigin()[:-1]
            spacing = mask_img.GetSpacing()[:-1]
            direction = tuple(el for i, el in enumerate(mask_img.GetDirection()[:12]) if not (i + 1) % 4 == 0)
            size = mask_img.GetSize()[:-1]

        # assert meta-info roi == meta-info pet
        assert pet_img.GetOrigin() == origin
        assert pet_img.GetSpacing() == spacing
        assert pet_img.GetDirection() == direction
        assert pet_img.GetSize() == size

        # generate mask from ROIs
        new_mask = np.zeros(mask_array.shape[1:], dtype=np.int8)

        for num_slice in range(mask_array.shape[0]):
            mask_slice = mask_array[num_slice]

            # calculate threshold value of the roi
            if threshold == 'auto':
                roi = pet_array[mask_slice > 0]
                if len(roi) > 0:
                    SUV_max = np.max(roi)
                    threshold_suv = SUV_max * 0.41
                else:
                    threshold_suv = 0.0
            elif threshold == 'otsu':
                roi = pet_array[mask_slice > 0]
                if len(roi) > 0:
                    try:
                        threshold_suv = filters.threshold_otsu(roi)
                    except:
                        threshold_suv = 0.0
                else:
                    threshold_suv = 0.0
            else:
                threshold_suv = threshold

            # apply threshold
            new_mask[np.where((pet_array > threshold_suv) & (mask_slice > 0))] = 1

        # reconvert to sitk and restore information
        new_mask = sitk.GetImageFromArray(new_mask)
        new_mask.SetOrigin(origin)
        new_mask.SetDirection(direction)
        new_mask.SetSpacing(spacing)

        return new_mask


class DataGenerator(tf.keras.utils.Sequence):
    """
    Read, preprocess and flow the PET scan, CT scan and mask

    """

    def __init__(self,
                 images_paths,
                 target_shape=None,
                 target_voxel_spacing=None,
                 augmentation=False,
                 normalize=True,
                 batch_size=1,
                 shuffle=True,
                 threshold='otsu'):
        """
        :param images_paths:         list of tuple : [{'pet_img': pet_path, 'ct_img': ct_path, 'mask_img': mask_path}, ...]
        :param batch_size:           int
        :param shuffle:              bool
        :param augmentation:         bool
        :param target_shape:         tuple, shape of generated PET, CT or MASK scan: (z, y, x) (368, 128, 128) for ex.
        :param target_voxel_spacing: tuple, resolution of the generated PET, CT or MASK scan : (z, y, x) (4.8, 4.8, 4.8) for ex.

        """
        self.images_paths = images_paths
        self.number_channels = 6  # PET + CT scan

        # self.images_shape = tuple(list(target_shape[1:]) + [self.number_channels])

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augmentation

        self.on_epoch_end()

        self.dtypes = {'pet_img': sitk.sitkFloat32,
                       'ct_img': sitk.sitkFloat32,
                       'mask_img': sitk.sitkUInt8}
        self.threshold = threshold
        if augmentation:
            augment = DataAugmentor(translation=(10, 10, 0), scaling=(0.1, 0.1, 0.0), rotation=(pi / 30, pi / 30, 0.0),
                                    default_value={'pet_img': 0.0, 'ct_img': -1000.0, 'mask_img': 0})
        else:
            augment = None

        self.preprocessor = InputPipeline(target_shape=target_shape,
                                          target_voxel_spacing=target_voxel_spacing,
                                          normalize=normalize,
                                          augment=augment)

    def __len__(self):
        """
        :return: int, the number of batches per epoch
        """
        return int(np.floor(len(self.images_paths)))

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
        # indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        img_path = self.images_paths[self.indexes[index]]

        # prepare the batch
        X_batch = []
        Y_batch = []

        # read images
        pet_path, ct_path, mask_path = img_path['pet_img'], img_path['ct_img'], img_path['mask_img']

        pet_img = self.read_PET(pet_path)
        ct_img = self.read_CT(ct_path)
        mask_img = self.read_MASK(mask_path)

        threshold = self.threshold
        images = self.preprocessor({'pet_img': pet_img, 'ct_img': ct_img, 'mask_img': mask_img},
                                   threshold=threshold)

        # convert to numpy array
        pet_array = sitk.GetArrayFromImage(images['pet_img'])
        ct_array = sitk.GetArrayFromImage(images['ct_img'])
        mask_array = sitk.GetArrayFromImage(images['mask_img'])

        n_slice = self.number_channels // 2
        for i in random.sample(range(0, pet_array.shape[0] + 1 - n_slice), self.batch_size):
            # select slices
            pet_ct_slices = np.vstack((pet_array[i:i + n_slice], ct_array[i:i + n_slice]))
            pet_ct_slices = np.transpose(pet_ct_slices, (1, 2, 0))
            mask_array_slice = mask_array[i + n_slice // 2]

            # add one channel
            mask_array_slice = np.expand_dims(mask_array_slice, axis=-1)

            X_batch.append(pet_ct_slices)
            Y_batch.append(mask_array_slice)

        return np.array(X_batch), np.array(Y_batch)

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


class FullPipeline(object):
    """
    Read, preprocess and flow the PET scan, CT scan and mask

    """

    def __init__(self,
                 model,
                 target_shape=None,
                 target_voxel_spacing=None,
                 normalize=True,
                 in_channels=6,
                 threshold='otsu'):
        """
        :param target_shape:         tuple, shape of generated PET, CT or MASK scan: (z, y, x) (368, 128, 128) for ex.
        :param target_voxel_spacing: tuple, resolution of the generated PET, CT or MASK scan : (z, y, x) (4.8, 4.8, 4.8) for ex.

        """
        self.target_shape = target_shape
        self.target_voxel_sapcing = target_voxel_spacing
        self.number_channels = in_channels  # PET + CT scan
        # self.images_shape = tuple(list(target_shape[1:]) + [self.number_channels])

        self.preprocessor = InputPipeline(target_shape=target_shape,
                                          target_voxel_spacing=target_voxel_spacing,
                                          normalize=normalize,
                                          augment=False)

        self.dtypes = {'pet_img': sitk.sitkFloat32,
                       'ct_img': sitk.sitkFloat32,
                       'mask_img': sitk.sitkUInt8}
        self.threshold = threshold

        self.model = self.load_model(model) if isinstance(model, str) else model

    def __call__(self, img_dict):
        pet_img, ct_img = img_dict['pet_img'], img_dict['ct_img']

        if isinstance(pet_img, str):
            pet_img = self.read_PET(pet_img)
        if isinstance(ct_img, str):
            ct_img = self.read_CT(ct_img)

        if 'mask_img' in img_dict:
            mask_img = img_dict['mask_img']
            if isinstance(mask_img, str):
                mask_img = self.read_MASK(mask_img)

        # read original meta info
        origin = pet_img.GetOrigin()
        spacing = pet_img.GetSpacing()
        direction = pet_img.GetDirection()
        size = pet_img.GetSize()

        # preprocess inputs
        if 'mask_img' in img_dict:
            original_mask = self.preprocessor.roi2mask(mask_img, pet_img,
                                                       threshold=self.threshold)

            images = self.preprocessor({'pet_img': pet_img, 'ct_img': ct_img, 'mask_img': mask_img},
                                       threshold=self.threshold)

        else:
            images = self.preprocessor({'pet_img': pet_img, 'ct_img': ct_img})

        # convert to numpy array
        pet_array = sitk.GetArrayFromImage(images['pet_img'])
        ct_array = sitk.GetArrayFromImage(images['ct_img'])

        X_batch = []
        n_slice = self.number_channels // 2
        for i in range(0, pet_array.shape[0] + 1 - n_slice):
            # select slices
            pet_ct_slices = np.vstack((pet_array[i:i + n_slice], ct_array[i:i + n_slice]))
            pet_ct_slices = np.transpose(pet_ct_slices, (1, 2, 0))

            X_batch.append(pet_ct_slices)

        X_batch = np.array(X_batch)
        Y_batch = self.model.predic(X_batch)
        Y_batch = np.squeeze(Y_batch)

        n_zero_up = n_slice // 2
        n_zero_down = n_slice - 1 - n_zero_up
        mask_array = np.concatenate((np.zeros((n_zero_up, Y_batch.shape[2], Y_batch.shape[3])),
                                     Y_batch,
                                     np.zeros((n_zero_down, Y_batch.shape[2], Y_batch.shape[3]))), axis=0)

        # transform numpy array to simple itk image / nifti format
        mask_img = sitk.GetImageFromArray(mask_array)
        mask_img.SetOrigin(self.preprocessor.compute_new_origin_head2hip(pet_img))
        mask_img.SetDirection(self.preprocessor.target_direction)
        mask_img.SetSpacing(self.preprocessor.target_voxel_spacing)

        # resample to orginal shape, spacing, direction and origin
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(direction)
        transformation.SetOutputOrigin(origin)
        transformation.SetOutputSpacing(spacing)
        transformation.SetSize(size)

        transformation.SetDefaultPixelValue(0.0)
        transformation.SetInterpolator(sitk.sitkLinear)  # sitk.sitkNearestNeighbor
        mask_img_final = transformation.Execute(mask_img)

        # transform to binary
        mask_img_final = sitk.BinaryThreshold(mask_img_final, lowerThreshold=0.0, upperThreshold=0.5, insideValue=0, outsideValue=1)

        if 'mask_img' in img_dict:

            return {'mask_cnn_pred': mask_img, 'mask_cnn_true': images['mask_img'],
                    'mask_pred': mask_img_final, 'mask_true': original_mask}

        else:
            return mask_img_final

    @staticmethod
    def load_model(model_path):
        return tf.keras.models.load_model(model_path, compile=False)

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
