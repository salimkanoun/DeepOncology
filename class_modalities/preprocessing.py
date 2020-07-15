import numpy as np
import SimpleITK as sitk


class preprocessor(object):
    """
    preprocessor PET, CT, MASK scan
    """

    def __init__(self,
                 target_shape=None,
                 target_voxel_spacing=None,
                 resize=True,
                 normalize=True
                 ):

        self.target_shape = target_shape  # [x, y, z]
        self.target_voxel_spacing = target_voxel_spacing[::-1]
        self.target_direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        self.resize = resize
        self.default_value = {'PET': 0.0,
                              'CT': -1024.0,
                              'MASK': 0}
        self.normalize = normalize
        self.dtypes = {'PET': sitk.sitkFloat32,
                       'CT': sitk.sitkFloat32,
                       'mask': sitk.sitkUInt8}

    def transform(self, inputs, threshold=None):
        # read input
        pet_img, ct_img, mask_img = inputs['pet_img'], inputs['ct_img'], inputs['mask_img']

        if threshold is not None:
            # get mask from ROI
            mask_img = self.roi2mask(mask_img, pet_img, threshold=threshold)

        # normalize before resample
        if self.normalize:
            pet_img = self.normalize_PET(ct_img)
            ct_img = self.normalize_CT(ct_img)

        # resample to sample shape and spacing resolution
        pet_img, ct_img, mask_img = self.resample_PET_CT_MASK(pet_img, ct_img, mask_img)

        return {'pet_img': pet_img, 'ct_img': ct_img, 'mask_img': mask_img}

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

    def resample_PET(self, PET_img, new_Origin):
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(self.target_direction)
        transformation.SetOutputOrigin(new_Origin)
        transformation.SetOutputSpacing(self.target_voxel_spacing)
        transformation.SetSize(self.target_shape)

        transformation.SetDefaultPixelValue(self.default_value['PET'])
        transformation.SetInterpolator(sitk.sitkBSpline)

        return transformation.Execute(PET_img)

    def resample_CT(self, CT_img, new_Origin):
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(self.target_direction)
        transformation.SetOutputOrigin(new_Origin)
        transformation.SetOutputSpacing(self.target_voxel_spacing)
        transformation.SetSize(self.target_shape)

        transformation.SetDefaultPixelValue(self.default_value['CT'])
        transformation.SetInterpolator(sitk.sitkBSpline)

        return transformation.Execute(CT_img)

    def resample_MASK(self, MASK_img, new_origin):
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(self.target_direction)
        transformation.SetOutputOrigin(new_origin)
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
        new_origin = self.compute_new_Origin(PET_img)

        # apply transformation : resample and reshape
        PET_img = self.resample_PET(PET_img, new_origin)
        CT_img = self.resample_CT(CT_img, new_origin)
        MASK_img = self.resample_MASK(MASK_img, new_origin)

        return PET_img, CT_img, MASK_img

    def compute_new_Origin(self, PET_img):

        origin = np.asarray(PET_img.GetOrigin())
        shape = np.asarray(PET_img.GetSize())
        spacing = np.asarray(PET_img.GetSpacing())
        new_shape = np.asarray(self.target_shape)
        new_spacing = np.asarray(self.target_voxel_spacing)

        return tuple(origin + 0.5 * (shape * spacing - new_shape * new_spacing))

    @staticmethod
    def roi2mask(mask_img, pet_img, threshold='auto'):
        """
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
        else:
            origin = mask_img.GetOrigin()[:-1]
            spacing = mask_img.GetSpacing()[:-1]
            direction = tuple(el for i, el in enumerate(mask_img.GetDirection()[:12]) if not (i + 1) % 4 == 0)

        # assert meta-info roi == meta-info pet
        assert pet_img.GetOrigin() == origin
        assert pet_img.GetSpacing() == spacing
        assert pet_img.GetDirection() == direction
        assert pet_img.GetSize() == mask_img.GetSize()

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
            new_mask[np.where((pet_array > threshold_suv) & (mask_slice > 0))] = 1

        # reconvert to sitk and restore information
        new_mask = sitk.GetImageFromArray(new_mask)
        new_mask.SetOrigin(origin)
        new_mask.SetDirection(direction)
        new_mask.SetSpacing(spacing)

        return new_mask
