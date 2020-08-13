import numpy as np
import SimpleITK as sitk
from skimage import filters


class PreprocessorPETCT(object):
    """
    preprocessor PET, CT, MASK scan
    """

    def __init__(self,
                 target_shape=None,
                 target_voxel_spacing=None,
                 resize=True,
                 normalize=True,
                 origin='head'
                 ):

        self.target_shape = target_shape[::-1]  # [z, y, x] to [x, y, z]
        self.target_voxel_spacing = target_voxel_spacing[::-1]  # [z, y, x] to [x, y, z]
        self.target_direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        self.resize = resize
        self.default_value = {'pet_img': 0.0,  # after normalize
                              'ct_img': 0.0,
                              'mask_img': 0}
        self.normalize = normalize
        self.origin = origin  # 'head' or 'middle'
        self.dtypes = {'pet_img': sitk.sitkFloat32,
                       'ct_img': sitk.sitkFloat32,
                       'mask_img': sitk.sitkUInt8}

        self.interpolator = {'pet_img': sitk.sitkBSpline,
                             'ct_img': sitk.sitkBSpline,
                             'mask_img': sitk.sitkNearestNeighbor}

    def __call__(self, inputs, threshold=None):
        return self.transform(inputs, threshold)

    def transform(self, inputs, threshold=None):
        if len(inputs) == 3:
            # read input
            pet_img, ct_img, mask_img = inputs['pet_img'], inputs['ct_img'], inputs['mask_img']

            if threshold is not None:
                # get mask from ROI
                mask_img = self.roi2mask(mask_img, pet_img, threshold=threshold)

            # normalize before resample
            if self.normalize:
                pet_img = self.normalize_PET(pet_img)
                ct_img = self.normalize_CT(ct_img)
            # resample to sample shape and spacing resolution
            pet_img, ct_img, mask_img = self.resample_PET_CT_MASK(pet_img, ct_img, mask_img)

            return {'pet_img': pet_img, 'ct_img': ct_img, 'mask_img': mask_img}
        else:
            # read input
            pet_img, ct_img = inputs['pet_img'], inputs['ct_img']

            # normalize before resample
            if self.normalize:
                pet_img = self.normalize_PET(pet_img)
                ct_img = self.normalize_CT(ct_img)

            # resample to sample shape and spacing resolution
            pet_img, ct_img = self.resample_PET_CT(pet_img, ct_img)

            return {'pet_img': pet_img, 'ct_img': ct_img}

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

    def resample_PET(self, pet_img, new_origin):
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(self.target_direction)
        transformation.SetOutputOrigin(new_origin)
        transformation.SetOutputSpacing(self.target_voxel_spacing)
        transformation.SetSize(self.target_shape)

        transformation.SetDefaultPixelValue(self.default_value['pet_img'])
        transformation.SetInterpolator(sitk.sitkBSpline)

        return transformation.Execute(pet_img)

    def resample_CT(self, ct_img, new_origin):
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(self.target_direction)
        transformation.SetOutputOrigin(new_origin)
        transformation.SetOutputSpacing(self.target_voxel_spacing)
        transformation.SetSize(self.target_shape)

        transformation.SetDefaultPixelValue(self.default_value['ct_img'])
        transformation.SetInterpolator(sitk.sitkBSpline)

        return transformation.Execute(ct_img)

    def resample_MASK(self, MASK_img, new_origin):
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(self.target_direction)
        transformation.SetOutputOrigin(new_origin)
        transformation.SetOutputSpacing(self.target_voxel_spacing)
        transformation.SetSize(self.target_shape)

        transformation.SetDefaultPixelValue(self.default_value['mask_img'])
        transformation.SetInterpolator(sitk.sitkNearestNeighbor)

        return transformation.Execute(MASK_img)

    def resample_PET_CT(self, PET_img, CT_img):
        """
        resample and reshape PET, CT and MASK to the same origin, direction, spacing and shape
        """
        # compute transformation parameters
        new_origin = self.compute_new_origin(PET_img)

        # apply transformation : resample and reshape
        PET_img = self.resample_PET(PET_img, new_origin)
        CT_img = self.resample_CT(CT_img, new_origin)

        return PET_img, CT_img

    def resample_PET_CT_MASK(self, PET_img, CT_img, MASK_img):
        """
        resample and reshape PET, CT and MASK to the same origin, direction, spacing and shape
        """
        # compute transformation parameters
        new_origin = self.compute_new_origin(PET_img)

        # apply transformation : resample and reshape
        PET_img = self.resample_PET(PET_img, new_origin)
        CT_img = self.resample_CT(CT_img, new_origin)
        MASK_img = self.resample_MASK(MASK_img, new_origin)

        return PET_img, CT_img, MASK_img

    def resample_img(self, img, new_origin, default_value, interpolator):
        # transformation parametrisation
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(self.target_direction)
        transformation.SetOutputOrigin(new_origin)
        transformation.SetOutputSpacing(self.target_voxel_spacing)
        transformation.SetSize(self.target_shape)

        transformation.SetDefaultPixelValue(default_value)
        transformation.SetInterpolator(interpolator)

        return transformation.Execute(img)

    def resample_images(self, inputs):
        # compute transformation parameters
        new_origin = self.compute_new_origin(inputs['pet_img'])

        output = dict()
        for key, img in inputs:
            output[key] = self.resample_img(img, new_origin, self.default_value[key], self.interpolator[key])
        return output

    def compute_new_origin_head2hip(self, pet_img):
        new_shape = self.target_shape
        new_spacing = self.target_voxel_spacing
        pet_size = pet_img.GetSize()
        pet_spacing = pet_img.GetSpacing()
        pet_origin = pet_img.GetOrigin()
        new_origin = (pet_origin[0] + 0.5 * pet_size[0] * pet_spacing[0] - 0.5 * new_shape[0] * new_spacing[0],
                      pet_origin[1] + 0.5 * pet_size[1] * pet_spacing[1] - 0.5 * new_shape[1] * new_spacing[1],
                      pet_origin[2] + 1.0 * pet_size[2] * pet_spacing[2] - 1.0 * new_shape[2] * new_spacing[2])
        return new_origin

    def compute_new_origin_centered_img(self, pet_img):
        origin = np.asarray(pet_img.GetOrigin())
        shape = np.asarray(pet_img.GetSize())
        spacing = np.asarray(pet_img.GetSpacing())
        new_shape = np.asarray(self.target_shape)
        new_spacing = np.asarray(self.target_voxel_spacing)

        return tuple(origin + 0.5 * (shape * spacing - new_shape * new_spacing))

    def compute_new_origin(self, pet_img):
        if self.origin == 'middle':
            return self.compute_new_origin_centered_img(pet_img)
        elif self.origin == 'head':
            return self.compute_new_origin_head2hip(pet_img)

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

        # print(pet_img.GetOrigin(), origin)
        # print(pet_img.GetSpacing(), spacing)
        # print(pet_img.GetDirection(), direction)
        # print(pet_img.GetSize(), size)
        # print(mask_array.shape)

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
                    threshold_suv = filters.threshold_otsu(roi)
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
