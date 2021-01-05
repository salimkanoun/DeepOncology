import os
from lib.utils import read_cfg

import numpy as np
import tensorflow as tf
from .preprocessing import get_transform_test
from lib.transforms import PostCNNResampler


class Pipeline(object):

    def __init__(self, model_path=None):
        #self.cfg_path = cfg
        #self.load_cfg()

        self.build_transformers()
        self.model_path = model_path
        self.load_model()

    #def load_cfg(self):
    #    self.cfg = read_cfg(self.cfg_path)

    def load_model(self):
        if self.model_path is None:
            #folder_path = os.path.dirname(self.cfg_path)
            print("No model path to load")
            #self.model_path = os.path.join(folder_path, 'model_weights.h5')
        
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def build_transformers(self):
        modalities = ('pet_img', 'ct_img')
        self.pre_transforms = get_transform_test(modalities)
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



