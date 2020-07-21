import numpy as np
import SimpleITK as sitk

import os
import re


def resample_img(img,
                 target_direction, new_origin,
                 target_voxel_spacing, target_shape,
                 default_value, interpolator):

    transformation = sitk.ResampleImageFilter()
    transformation.SetOutputDirection(target_direction)
    transformation.SetOutputOrigin(new_origin)
    transformation.SetOutputSpacing(target_voxel_spacing)
    transformation.SetSize(target_shape)

    transformation.SetDefaultPixelValue(default_value)
    transformation.SetInterpolator(interpolator)

    return transformation.Execute(img)


def normalize_img(img, window_min, window_max):
    """
    Transform input value from window_min - window_max to 0 - 1
    """
    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    intensityWindowingFilter.SetOutputMaximum(1)
    intensityWindowingFilter.SetOutputMinimum(0)
    intensityWindowingFilter.SetWindowMaximum(window_max)
    intensityWindowingFilter.SetWindowMinimum(window_min)
    return intensityWindowingFilter.Execute(img)


def normalize_img_v2(img, shift, scale):
    return sitk.ShiftScale(img, shift=shift, scale=scale)


def threshold_img(img, threshold):
    return sitk.Threshold(img, lower=0.0, upper=threshold, outsideValue=threshold)


def mip(img, threshold=None):
    img_array = sitk.GetArrayFromImage(img)

    if threshold:
        # img_array = np.array(img_array>threshold, dtype=np.int8)
        img_array[img_array > threshold] = threshold
    return np.max(img_array, axis=1), np.max(img_array, axis=2)


def get_info(img):
    print('img information :')
    print('\t Origin    :', img.GetOrigin())
    print('\t Size      :', img.GetSize())
    print('\t Spacing   :', img.GetSpacing())
    print('\t Direction :', img.GetDirection())


def get_study_uid(img_path):
    return re.sub('_nifti_(PT|mask|CT)\.nii(\.gz)?', '', os.path.basename(img_path))
