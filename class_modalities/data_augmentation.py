import numpy as np
import SimpleITK as sitk

import random
from math import pi


class DataAugmentor(object):

    def __init__(self, translation=10, scaling=0.1, rotation=pi/30):

        self.translation = translation if isinstance(translation, tuple) else (translation, translation, translation)
        self.scaling = scaling if isinstance(scaling, tuple) else (scaling, scaling, scaling)
        self.rotation = rotation

        self.default_value = {'pet': 0, 'ct': 0, 'mask': 0}
        self.interpolator = {'pet': sitk.sitkBSpline, 'ct': sitk.sitkBSpline, 'mask': sitk.sitkNearestNeighbor}

    @staticmethod
    def generate_random_bool(p):
        """
        :param p : float between 0-1, probability
        :return: True if a probobility of p
        """
        return random.random() < p

    def generate_random_deformation_ratios(self):
        """
        :return: dict with random deformation
        """

        deformation = dict()
        if self.generate_random_bool(0.8):
            deformation['translation'] = (random.uniform(-1.0*self.translation[0], self.translation[0]),
                                          random.uniform(-1.0*self.translation[1], self.translation[1]),
                                          random.uniform(-1.0*self.translation[2], self.translation[2]))
        else:
            deformation['translation'] = (0, 0, 0)

        if self.generate_random_bool(0.8):
            deformation['scaling'] = (random.uniform(1.0 - self.scaling[0], 1.0 + self.scaling[0]),
                                      random.uniform(1.0 - self.scaling[1], 1.0 + self.scaling[1]),
                                      random.uniform(1.0 - self.scaling[2], 1.0 + self.scaling[2]))
        else:
            deformation['scaling'] = (1.0, 1.0, 1.0)

        if self.generate_random_bool(0.8):
            deformation['rotation'] = random.uniform(-1.0*self.rotation, self.rotation)
        else:
            deformation['rotation'] = 0.0

        return deformation

    @staticmethod
    def AffineTransformation(image, interpolator, deformations, default_value):
        """
        Apply deformation to the input image
        :parameter
            :param image: PET, CT or MASK
            :param interpolator: method of interpolator, for ex : sitk.sitkBSpline
            :param deformations: dict of deformation to apply
            :param default_value: default value to fill the image
        :return: deformed image
        """

        center = tuple(
            np.asarray(image.GetOrigin()) + 0.5 * np.asarray(image.GetSize()) * np.asarray(image.GetSpacing()))

        transformation = sitk.AffineTransform(3)
        transformation.SetCenter(center)
        transformation.Scale(deformations['scaling'])
        transformation.Rotate(axis1=0, axis2=2, angle=deformations['rotation'])
        transformation.Translate(deformations['translation'])
        reference_image = image

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
        def_ratios = self.generate_random_deformation_ratios()

        # apply deformation
        new_PET_img = self.AffineTransformation(image=PET_img,
                                                interpolator=self.interpolator['pet'],
                                                deformations=def_ratios,
                                                default_value=self.default_value['pet'])
        new_CT_img = self.AffineTransformation(image=CT_img,
                                               interpolator=self.interpolator['ct'],
                                               deformations=def_ratios,
                                               default_value=self.default_value['ct'])
        new_MASK_img = self.AffineTransformation(image=MASK_img,
                                                 interpolator=self.interpolator['mask'],
                                                 deformations=def_ratios,
                                                 default_value=self.default_value['mask'])
        return new_PET_img, new_CT_img, new_MASK_img
