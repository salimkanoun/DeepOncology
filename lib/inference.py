import numpy as np
import SimpleITK as sitk

from lib.transforms import *

import tensorflow as tf

import os
import json


class Pipeline(object):

    def __init__(self, cfg, model_path=None):
        self.cfg_path = cfg

        self.load_cfg()
        self.model_path = model_path
        self.load_model()
        self.build_transformers()

    def load_cfg(self):
        with open(self.cfg_path) as f:
            self.config = json.load(f)

    def load_model(self):

        if self.model_path is None:
            folder_path = os.path.dirname(self.cfg_path)
            self.model_path = os.path.join(folder_path, 'model_weights.h5')

        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def build_transformers(self):
        image_shape = tuple(self.config['preprocessing']['image_shape'])
        voxel_spacing = tuple(self.config['preprocessing']['voxel_spacing'])

        self.pre_transforms = Compose([LoadNifti(keys=("pet_img", "ct_img")),
                                       ResampleReshapeAlign(target_shape=image_shape[::-1],
                                                            target_voxel_spacing=voxel_spacing[::-1],
                                                            keys=('pet_img', "ct_img"),
                                                            origin='head', origin_key='pet_img',
                                                            interpolator={'pet_img': sitk.sitkLinear,
                                                                          'ct_img': sitk.sitkLinear},
                                                            default_value={'pet_img': 0.0,
                                                                           'ct_img': -1000.0}),
                                       Sitk2Numpy(keys=['pet_img', 'ct_img']),
                                       # normalize input
                                       ScaleIntensityRanged(keys="pet_img",
                                                            a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True),
                                       ScaleIntensityRanged(keys="ct_img",
                                                            a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0,
                                                            clip=True),
                                       ConcatModality(keys=('pet_img', 'ct_img'), channel_first=False, new_key='input')
                                       ])

        self.post_transforms = PostCNNResampler(0.5)

    def __call__(self, img_path):
        return self.predict(img_path)

    def predict(self, img_path):
        """
        img_path dict :
        {'pet_img': '/abs/path/to/nifti_PT.nii',
        'ct_img': '/abs/path/to/nifti_CT.nii'}

        :return
            Simple ITK binary mask prediction.
         """
        images_dict = self.pre_transforms(img_path)

        img_prepro = images_dict['input']
        input_cnn = np.expand_dims(img_prepro, axis=0)

        mask_pred = self.model.predict(input_cnn)
        mask_pred = np.squeeze(mask_pred)

        images_dict['mask_pred'] = mask_pred
        mask_pred = self.post_transforms(images_dict)

        return mask_pred



