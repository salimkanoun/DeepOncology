import SimpleITK as sitk
import numpy as np

from skimage.measure import label


def vol_lymphoma(mask_img):
    spacing = mask_img.GetSpacing()
    vol_spacing = 1.0
    for el in spacing:
        vol_spacing *= el

    mask_array = sitk.GetArrayFromImage(mask_img)
    return np.sum(mask_array) * vol_spacing


def get_lymphoma(mask_img):
    mask_array = sitk.GetArrayFromImage(mask_img)
    return label(mask_array)


def ann_arbor_classification(mask_img):
    pass