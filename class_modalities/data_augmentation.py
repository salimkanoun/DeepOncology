import numpy as np
import SimpleITK as sitk

import random
from math import pi


class DataAugmentor(object):

    def __init__(self, translation=10, scaling=0.1, rotation=(0.0, pi/30, 0.0),
                 default_value=None, interpolator=None):

        if interpolator is None:
            interpolator = {'pet_img': sitk.sitkBSpline, 'ct_img': sitk.sitkBSpline,
                            'mask_img': sitk.sitkNearestNeighbor}
        if default_value is None:
            default_value = {'pet_img': 0.0, 'ct_img': 0.0, 'mask_img': 0}

        self.translation = translation if isinstance(translation, tuple) else (translation, translation, translation)
        self.scaling = scaling if isinstance(scaling, tuple) else (scaling, scaling, scaling)
        self.rotation = rotation if isinstance(rotation, tuple) else (rotation, rotation, rotation)

        self.default_value = default_value
        self.interpolator = interpolator

    def __call__(self, inputs):
        pet_img, ct_img, mask_img = inputs['pet_img'], inputs['ct_img'], inputs['mask_img']
        pet_img, ct_img, mask_img = self.data_augmentor(pet_img, ct_img, mask_img)
        return {'pet_img': pet_img, 'ct_img': ct_img, 'mask_img': mask_img}

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
            deformation['translation'] = (random.uniform(-1.0 * self.translation[0], self.translation[0]),
                                          random.uniform(-1.0 * self.translation[1], self.translation[1]),
                                          random.uniform(-1.0 * self.translation[2], self.translation[2]))
        else:
            deformation['translation'] = (0, 0, 0)

        if self.generate_random_bool(0.8):
            deformation['scaling'] = (random.uniform(1.0 - self.scaling[0], 1.0 + self.scaling[0]),
                                      random.uniform(1.0 - self.scaling[1], 1.0 + self.scaling[1]),
                                      random.uniform(1.0 - self.scaling[2], 1.0 + self.scaling[2]))
        else:
            deformation['scaling'] = (1.0, 1.0, 1.0)

        if self.generate_random_bool(0.8):
            deformation['rotation'] = (random.uniform(-1.0 * self.rotation[0], self.rotation[0]),
                                       random.uniform(-1.0 * self.rotation[1], self.rotation[1]),
                                       random.uniform(-1.0 * self.rotation[2], self.rotation[2]))
        else:
            deformation['rotation'] = (0.0, 0.0, 0.0)

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
        transformation.Rotate(axis1=1, axis2=2, angle=deformations['rotation'][0])
        transformation.Rotate(axis1=0, axis2=2, angle=deformations['rotation'][1])
        transformation.Rotate(axis1=0, axis2=1, angle=deformations['rotation'][2])
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
        new_pet_img = self.AffineTransformation(image=PET_img,
                                                interpolator=self.interpolator['pet_img'],
                                                deformations=def_ratios,
                                                default_value=self.default_value['pet_img'])
        new_ct_img = self.AffineTransformation(image=CT_img,
                                               interpolator=self.interpolator['ct_img'],
                                               deformations=def_ratios,
                                               default_value=self.default_value['ct_img'])
        new_mask_img = self.AffineTransformation(image=MASK_img,
                                                 interpolator=self.interpolator['mask_img'],
                                                 deformations=def_ratios,
                                                 default_value=self.default_value['mask_img'])
        return new_pet_img, new_ct_img, new_mask_img
