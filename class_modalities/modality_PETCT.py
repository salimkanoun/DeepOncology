import numpy as np
import SimpleITK as sitk

import random
from math import pi

import tensorflow as tf


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
                 normalize=True):
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

        self.target_shape = target_shape[::-1]  # [z, y, x] to [
        self.target_voxel_spacing = target_voxel_spacing[::-1]
        self.resize = resize
        self.default_value = {'PET': 0.0,
                              'CT': -1024.0,
                              'MASK': 0}
        self.normalize = normalize
        self.dtypes = {'PET': sitk.sitkFloat32,
                       'CT': sitk.sitkFloat32,
                       'mask': sitk.sitkUInt8}

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

            # load and resample images
            PET_img, CT_img, MASK_img = self.preprocess_data(PET_id, CT_id, MASK_id)

            if self.augment:
                PET_img, CT_img, MASK_img = self.data_augmentor(PET_img, CT_img, MASK_img)

            # convert to numpy array
            PET_array = sitk.GetArrayFromImage(PET_img)
            CT_array = sitk.GetArrayFromImage(CT_img)
            MASK_img = sitk.GetArrayFromImage(MASK_img)

            # normalize data
            if self.normalize:
                PET_array = self.normalize_PET(PET_array)
                CT_array = self.normalize_CT(CT_array)

            # concatenate PET and CT
            PET_CT_array = np.stack((PET_array, CT_array), axis=-1)

            # add it to the batch
            X_batch.append(PET_CT_array)
            Y_batch.append(MASK_img)

        return np.array(X_batch), np.array(Y_batch)

    def preprocess_data(self, PET_id, CT_id, MASK_id):
        """
        :param PET_id: string, path to PET scan
        :param CT_id: string, path to CT scan
        :param MASK_id: string, path to MASK
        :return: return preprocessed PET, CT, MASK img
        """

        PET_img = self.read_PET(PET_id)
        CT_img = self.read_CT(CT_id)
        MASK_img = self.read_mask(MASK_id)

        # transform to 3D binary mask
        MASK_img = self.preprocess_MASK_4D(MASK_img, PET_img, threshold='auto')

        return self.resample_PET_CT_MASK(PET_img, CT_img, MASK_img)

    def read_PET(self, filename):
        return sitk.ReadImage(filename, self.dtypes['PET'])

    def read_CT(self, filename):
        return sitk.ReadImage(filename, self.dtypes['CT'])

    def read_mask(self, filename):
        return sitk.ReadImage(filename, self.dtypes['mask'])

    def preprocess_MASK_4D(self, mask_img, pet_img, threshold='auto'):
        """
        :param mask_img: sitk image, raw mask
        :param pet_img: sitk image, the corresponding pet scan
        :param threshold: threshold to apply to the ROI to get the tumor segmentation.
                if set to 'auto', it will take 42% of the maximum
        :return: sitk image, the ground truth segmentation
        """
        mask_array = sitk.GetArrayFromImage(mask_img)
        pet_array = sitk.GetArrayFromImage(pet_img)

        new_mask = np.zeros(mask_array.shape[1:], dtype=np.int8)

        for num_slice in range(mask_array.shape[0]):

            mask_slice = mask_array[num_slice]

            if threshold == 'auto':
                SUV_max = np.max(pet_array[mask_slice > 0])
                threshold_suv = SUV_max * 0.42
            else:
                threshold_suv = threshold

            new_mask[np.where((pet_array > threshold_suv) & (mask_slice > 0))] = 1

        # reconvert to sitk and restore information
        new_mask = sitk.GetImageFromArray(new_mask)
        direction = tuple(el for i, el in enumerate(mask_img.GetDirection()[:12]) if not (i + 1) % 4 == 0)
        new_mask.SetOrigin(mask_img.GetOrigin()[:-1])
        new_mask.SetDirection(direction)
        new_mask.SetSpacing(mask_img.GetSpacing()[:-1])

        return new_mask


    def preprocess_MASK_3D(self, mask_img):
        # transform to binary
        return sitk.Threshold(mask_img, lower=0.0, upper=1.0, outsideValue=1.0)


    def normalize_PET(self, PET_array):
        return PET_array/10.0
        # return sitk.ShiftScale(PET_img, shift=0.0, scale=1. / 10.)

    def normalize_CT(self, CT_array):
        CT_array[CT_array < -1024] = -1024.0
        CT_array[CT_array > 3000] = 3000.0

        return (CT_array + 1000.0)/2000.0
        # return sitk.ShiftScale(CT_img, shift=1000, scale=1. / 2000.)

    def resample_PET(self, PET_img, new_Origin, new_Direction):
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(new_Direction)
        transformation.SetOutputOrigin(new_Origin)
        transformation.SetOutputSpacing(self.target_voxel_spacing)
        transformation.SetSize(self.target_shape)

        transformation.SetDefaultPixelValue(self.default_value['PET'])
        transformation.SetInterpolator(sitk.sitkBSpline)

        return transformation.Execute(PET_img)

    def resample_CT(self, CT_img, new_Origin, new_Direction):
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(new_Direction)
        transformation.SetOutputOrigin(new_Origin)
        transformation.SetOutputSpacing(self.target_voxel_spacing)
        transformation.SetSize(self.target_shape)

        transformation.SetDefaultPixelValue(self.default_value['CT'])
        transformation.SetInterpolator(sitk.sitkBSpline)

        return transformation.Execute(CT_img)

    def resample_MASK(self, MASK_img, new_Origin, new_Direction):
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(new_Direction)
        transformation.SetOutputOrigin(new_Origin)
        transformation.SetOutputSpacing(self.target_voxel_spacing)
        transformation.SetSize(self.target_shape)

        transformation.SetDefaultPixelValue(self.default_value['MASK'])
        transformation.SetInterpolator(sitk.sitkNearestNeighbor)

        return transformation.Execute(MASK_img)

    def resample_PET_CT_MASK(self, PET_img, CT_img, MASK_img):
        """
        resample and reshape PET, CT and MASK to the same origin, direction, spacing and shape
        """
        # compute transformation parameters
        new_Origin = self.compute_new_Origin(PET_img)
        new_Direction = PET_img.GetDirection()

        # apply transformation : resample and reshape
        PET_img = self.resample_PET(PET_img, new_Origin, new_Direction)
        CT_img = self.resample_CT(CT_img, new_Origin, new_Direction)
        MASK_img = self.resample_MASK(MASK_img, new_Origin, new_Direction)

        return PET_img, CT_img, MASK_img

    def compute_new_Origin(self, PET_img):

        origin = np.asarray(PET_img.GetOrigin())
        shape = np.asarray(PET_img.GetSize())
        spacing = np.asarray(PET_img.GetSpacing())
        new_shape = np.asarray(self.target_shape)
        new_spacing = np.asarray(self.target_voxel_spacing)

        return tuple(origin + 0.5 * (shape * spacing - new_shape * new_spacing))

    def save_PET(self, PET_img, filename):
        sitk.WriteImage(PET_img, filename)

    def save_CT(self, CT_img, filename):
        sitk.WriteImage(CT_img, filename)

    def save_MASK(self, MASK_img, filename):
        sitk.WriteImage(MASK_img, filename)

    def generate_random_DeformationRatios(self):
        """
        :return: dict with random deformation
        """

        deformation = {'Translation': (random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)),
                       'Scaling': (random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)),
                       'Rotation': random.uniform((-pi / 30), (pi / 30))}
        return deformation

    def AffineTransformation(self, image, interpolator, deformations):
        """
        Apply deformation to the input image
        :param image: PET, CT or MASK
        :param interpolator: method of interpolator, for ex : sitk.sitkBSpline
        :param deformations: dict of deformation to apply
        :return: deformed image
        """

        center = tuple(
            np.asarray(image.GetOrigin()) + 0.5 * np.asarray(image.GetSize()) * np.asarray(image.GetSpacing()))

        transformation = sitk.AffineTransform(3)
        transformation.SetCenter(center)
        transformation.Scale(deformations['Scaling'])
        transformation.Rotate(axis1=0, axis2=2, angle=deformations['Rotation'])
        transformation.Translate(deformations['Translation'])
        reference_image = image
        default_value = 0.0

        return sitk.Resample(image, reference_image, transformation, interpolator, default_value)

    def data_augmentor(self, PET_img, CT_img, MASK_img):
        """
        Apply the same random deformation to PET, CT and MASK
        :param PET_img:
        :param CT_img:
        :param MASK_img:
        :return: PET, CT, MASK
        """

        # generate deformation
        def_ratios = self.generate_random_DeformationRatios()

        # apply deformation
        new_PET_img = self.AffineTransformation(image=PET_img,
                                                interpolator=sitk.sitkBSpline,
                                                deformations=def_ratios)
        new_CT_img = self.AffineTransformation(image=CT_img,
                                               interpolator=sitk.sitkBSpline,
                                               deformations=def_ratios)
        new_MASK_img = self.AffineTransformation(image=MASK_img,
                                                 interpolator=sitk.sitkNearestNeighbor,
                                                 deformations=def_ratios)
        return new_PET_img, new_CT_img, new_MASK_img
