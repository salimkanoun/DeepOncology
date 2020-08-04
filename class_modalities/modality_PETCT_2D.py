import tensorflow as tf

import numpy as np
import SimpleITK as sitk

from scipy.stats import truncnorm
import random

from .preprocessing import PreprocessorPETCT
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

    def compute_new_origin_head2hip(self, pet_img):
        new_shape = self.target_shape
        new_spacing = self.target_voxel_spacing
        pet_size = pet_img.GetSize()
        pet_spacing = pet_img.GetSpacing()
        pet_origin = pet_img.GetOrigin()
        height = min(pet_size[2] * pet_spacing[2], 1228.8)  # 256*4.8 = 1228.8 mm
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
        z_size, z_spacing = inputs_img['pet_img'].GetSize()[2], inputs_img['pet_img'].GetSpacing()[2]
        height = min(z_size * z_spacing, 1228.8)  # 256*4.8 = 1228.8 mm
        target_shape = (self.target_shape[0], self.target_shape[1], height/self.target_voxel_spacing[2])

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
            else:
                threshold_suv = threshold

            # apply threshold
            new_mask[np.where((pet_array >= threshold_suv) & (mask_slice > 0))] = 1

        # reconvert to sitk and restore information
        new_mask = sitk.GetImageFromArray(new_mask)
        new_mask.SetOrigin(origin)
        new_mask.SetDirection(direction)
        new_mask.SetSpacing(spacing)

        return new_mask


class DataGenerator(object):
    """
    Read, preprocess and flow the PET scan, CT scan and mask

    """

    def __init__(self,
                 images_paths,
                 batch_size=1,
                 shuffle=True,
                 augmentation=False,
                 target_shape=None,
                 target_voxel_spacing=None):
        """
        :param images_paths:         list of tuple : [({'pet_img': pet_path, 'ct_img': ct_path, 'mask_img': mask_path), ...]
        :param batch_size:           int
        :param shuffle:              bool
        :param augmentation:         bool
        :param target_shape:         tuple, shape of generated PET, CT or MASK scan: (z, y, x) (368, 128, 128) for ex.
        :param target_voxel_spacing: tuple, resolution of the generated PET, CT or MASK scan : (z, y, x) (4.8, 4.8, 4.8) for ex.

        """
        self.images_paths = images_paths
        self.number_channels = 6  # PET + CT scan

        self.images_shape = target_shape[1:]  # 2D target shape

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augmentation

        self.dtypes = {'pet_img': sitk.sitkFloat32,
                       'ct_img': sitk.sitkFloat32,
                       'mask_img': sitk.sitkUInt8}
        self.default_threshold = 'auto'
        self.preprocessor = InputPipeline(target_shape=target_shape,
                                          target_voxel_spacing=target_voxel_spacing,
                                          normalize=True,
                                          augment=True)

    def _genetator(self):

        for img_path in self.images_paths:
            # read images
            pet_path, ct_path, mask_path = img_path['pet_img'], img_path['ct_img'], img_path['mask_img']

            pet_img = self.read_PET(pet_path)
            ct_img = self.read_CT(ct_path)
            mask_img = self.read_MASK(mask_path)

            if self.augment:
                threshold = self.generate_random_threshold()
            else:
                threshold = self.default_threshold

            images = self.preprocessor({'pet_img': pet_img, 'ct_img': ct_img, 'mask_img': mask_img},
                                       threshold=threshold)
            # output['pet_img'], output['ct_img'], output['mask_img']

            if self.augment:
                images = self.data_augmentor(images)

            if self.normalize:
                images['pet_img'] = self.normalize_PET(images['pet_img'])
                images['ct_img'] = self.normalize_CT(images['ct_img'])

            # convert to numpy array
            pet_array = sitk.GetArrayFromImage(images['pet_img'])
            ct_array = sitk.GetArrayFromImage(images['ct_img'])
            mask_array = sitk.GetArrayFromImage(images['mask_img'])

            n_slice = self.number_channels//2
            for i in range(0, pet_array.shape[0] - n_slice):
                # select slices
                pet_ct_slices = np.stack((pet_array[i:i + n_slice], ct_array[i:i + n_slice]), axis=-1)
                mask_array_slice = mask_array[i + n_slice//2]

                # add one channel
                mask_array_slice = np.expand_dims(mask_array_slice, axis=-1)

                yield pet_ct_slices, mask_array_slice

    def get_dataset(self):
        return tf.data.Dataset.from_generator(
            self._genetator,
            (tf.float32, tf.int8),
            (tf.TensorShape((list(self.images_shape), list(self.images_shape))))
        )

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