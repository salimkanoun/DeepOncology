
import os
import shutil
from shutil import copyfile
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
import time
from tf.project.segmentation.lib.transforms import *
from tf.project.segmentation.lib.datasets import *
from dicom_to_cnn.tools.pre_processing.array_convertor import *
import time 


def get_transform(subset, modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin = None,  data_augmentation=True):
    """[summary]

    Args:
        
        subset ([str]): [train, val or test]
        modalities ([tuple]): [('pet_img, ct_img') or ('pet_img')]
        mode ([str]): [binary or probs]
        method ([str]): [ if binary, choose between : relative, absolute or otsu
                            else : method = []  ]
        tval ([float]) : [if mode = binary & method = relative : 0.41
                          if mode = binary & method = absolute : 2.5, 
                          else : don't need tval, tval = 0.0 ]

    """
    
    keys = tuple(list(modalities) + ['mask_img'])
    transformers = [LoadNifti(keys=keys)]  # Load NIFTI file from path

    # Generate ground-truth from PET and ROIs
    if mode == 'binary':
        transformers.append(Roi2Mask(keys=('pet_img', 'mask_img'),
                                         method=method, tval=tval))
    else : 
        transformers.append(Roi2MaskProbs(keys=('pet_img', 'mask_img'), new_key_name='mask_img'))

    transformers.append(ResampleReshapeAlign(target_size, target_spacing, target_direction, target_origin=None, keys=("pet_img", "ct_img", "mask_img"), test = False))

    if subset == 'train' and data_augmentation:
        translation = 10
        scaling = 0.1
        rotation = (np.pi / 60, np.pi / 30, np.pi / 60)
        transformers.append(RandAffine(keys=('pet_img', 'ct_img', 'mask_img'), translation=translation, scaling=scaling, rotation=rotation))
    
    # Convert Simple ITK image into numpy 3d-array
    transformers.append(DictSitk2Numpy(keys=('pet_img', 'ct_img', 'mask_img')))
    # Normalize input values
    
    for modality in modalities:
        if modality == 'pet_img' : 
            modal_pp = dict(a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True)
        else : 
            modal_pp = dict(a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True)

        transformers.append(ScaleIntensityRanged(keys = modality,
                                                 a_min =modal_pp['a_min'], a_max = modal_pp['a_max'], b_min=modal_pp['b_min'], b_max=modal_pp['b_max'], clip=modal_pp['clip']))

    # Concatenate modalities if necessary
    if len(modalities) > 1:
        transformers.append(ConcatModality(keys=modalities, channel_first=False, new_key='input'))
    else:
        transformers.append(AddChannel(keys=modalities, channel_first=False))
        transformers.append(RenameDict(keys=modalities, keys2='input'))

    transformers.append(AddChannel(keys='mask_img', channel_first=False))
    transformers = Compose(transformers)
    return transformers


def get_transform_test(modalities, target_size, target_spacing, target_direction, target_origin=None):
    """transformers for test set 

    Args:
        modalities ([tuple]): [('pet_img, ct_img') or ('pet_img')]

    Returns:
        [type]: [description]
    """
    keys = tuple(list(modalities))
    transformers = [LoadNifti(keys=keys)]  # Load NIFTI file from path
    transformers.append(ResampleReshapeAlign(target_size, target_spacing, target_direction, target_origin=None, keys=("pet_img", "ct_img", "mask_img"), test = True))
    transformers.append(DictSitk2Numpy(keys=('pet_img', 'ct_img', 'mask_img')))
    # Normalize input values
    for modality in modalities:
        if modality == 'pet_img' : 
            modal_pp = dict(a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True)
        else : 
            modal_pp = dict(a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True)
        #A METTRE EN PARAMETRES

        transformers.append(ScaleIntensityRanged(keys = modality,
                                                 a_min =modal_pp['a_min'], a_max = modal_pp['a_max'], b_min=modal_pp['b_min'], b_max=modal_pp['b_max'], clip=modal_pp['clip']))

    # Concatenate modalities if necessary
    if len(modalities) > 1:
        transformers.append(ConcatModality(keys=modalities, channel_first=False, new_key='input'))
    else:
        transformers.append(AddChannel(keys=modalities, channel_first=False))
        transformers.append(RenameDict(keys=modalities, keys2='input'))

    #transformers.append(AddChannel(keys='mask_img', channel_first=False))
    transformers = Compose(transformers)
    return transformers



#FUNCTION USE IN TRAINING SCRIPT TO GET DATA 
def get_data(pp_dir, csv_path, modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin=None , data_augmentation=True):
    """Save or not the pre processed data at nifti format

    Returns dataset, and list of transformers for train and val set. 
    """

        # Get Data
    DM = DataManager(csv_path=csv_path)
    train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)
    dataset = dict()
    dataset['train'], dataset['val'], dataset['test'] = train_images_paths, val_images_paths, test_images_paths

     # Define generator
    train_transforms = get_transform('train', modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin , data_augmentation = True,)
    val_transforms = get_transform('val', modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin ,  data_augmentation = False)

    return dataset, train_transforms, val_transforms


