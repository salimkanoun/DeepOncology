import SimpleITK as sitk
import tensorflow as tf
import numpy as np
import os
import re

from .preprocessing import PreprocessorPETCT as pp


class Pipeline(object):
    """
    Pipeline for prediction lymphoma segmentation
    """

    def __init__(self,
                 model_path,
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

        self.dtypes = {'pet': sitk.sitkFloat32,
                       'ct': sitk.sitkFloat32,
                       'mask': sitk.sitkUInt8}

        self.model = self.load_model(model_path)
        self.preprocessor = self.load_preprocessor()

    @staticmethod
    def load_model(model_path):
        return tf.keras.models.load_model(model_path, compile=False)

    def load_preprocessor(self):
        return pp(target_shape=self.target_shape,
                  target_voxel_spacing=self.target_voxel_spacing,
                  resize=self.resize,
                  normalize=self.normalize)

    def predict(self, inputs):
        """
        predict the mask from pet and ct scan
        Args :
            :param inputs: dict, {'pet_img': simple itk image, 'ct_img': simple itk image}
        :return: the predicted mask, simple itk image
        """
        pet_path, ct_path = inputs['pet_img'], inputs['ct_img']

        pet_img = self.read_PET(pet_path)
        ct_img = self.read_CT(ct_path)

        # read original meta info
        origin = pet_img.GetOrigin()
        spacing = pet_img.GetSpacing()
        direction = pet_img.GetDirection()
        size = pet_img.GetSize()

        # preprocess inputs
        data = self.preprocessor({'pet_img': pet_img, 'ct_img': ct_img})
        PET_array = sitk.GetArrayFromImage(data['pet_img'])
        CT_array = sitk.GetArrayFromImage(data['ct_img'])
        cnn_input = np.stack((PET_array, CT_array), axis=-1)

        # predict
        mask_array = self.model.predict(cnn_input)

        # postprocess mask
        # transform to binary
        # mask_array = np.round(mask_array)

        # transform numpy array to simple itk image / nifti format
        mask_img = sitk.GetImageFromArray(mask_array)
        mask_img.SetOrigin(pp.compute_new_Origin(pet_img))
        mask_img.SetDirection(self.target_direction)
        mask_img.SetSpacing(self.target_voxel_spacing)

        # resample to orginal shape, spacing, direction and origin
        transformation = sitk.ResampleImageFilter()
        transformation.SetOutputDirection(direction)
        transformation.SetOutputOrigin(origin)
        transformation.SetOutputSpacing(spacing)
        transformation.SetSize(size)

        transformation.SetDefaultPixelValue(0.0)
        transformation.SetInterpolator(sitk.sitkLinear)  # sitk.sitkNearestNeighbor
        mask_img = transformation.Execute(mask_img)

        # transform to binary
        mask_img = sitk.BinaryThreshold(mask_img, lowerThreshold=0.0, upperThreshold=0.5, insideValue=0, outsideValue=1)

        return mask_img

    def predict_on_batch(self, inputs, output_folder=''):
        """
        Args:
            :param inputs: list of dict: [{'pet_img': path/to/xxx.nii, 'ct_img': path/to/xxx.nii}, {}, ...]
            :param output_folder: path to folder to save result

        :return: None
        """
        for input in inputs:
            pet_path, ct_path = input['pet_img'], input['ct_img']
            study_uid = re.sub('_nifti_PT\.nii(\.gz)?', '', os.path.basename(pet_path))

            pet_img = self.read_PET(pet_path)
            ct_img = self.read_CT(ct_path)

            mask_pred = self.predict({'pet_img': pet_img, 'ct_img': ct_img})

            filename = os.path.join(output_folder, study_uid + '_nifti_mask_pred.nii')
            self.save_MASK(mask_pred, filename)

    def read_PET(self, filename):
        return sitk.ReadImage(filename, self.dtypes['pet'])

    def read_CT(self, filename):
        return sitk.ReadImage(filename, self.dtypes['ct'])

    def read_mask(self, filename):
        return sitk.ReadImage(filename, self.dtypes['mask'])

    def save_PET(self, PET_img, filename):
        sitk.WriteImage(PET_img, filename)

    def save_CT(self, CT_img, filename):
        sitk.WriteImage(CT_img, filename)

    def save_MASK(self, MASK_img, filename):
        sitk.WriteImage(MASK_img, filename)
