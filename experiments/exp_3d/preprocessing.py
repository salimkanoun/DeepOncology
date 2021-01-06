
import os
import shutil
from shutil import copyfile

import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
import time

from lib.datasets import DataManager
from lib.transforms import *

from lib.utils import sec2str

#put modalities in paramters 




def check_path_is_valid(files):
    indexes = files.keys() if isinstance(files, dict) else np.arange(len(files))

    for idx in indexes:
        for key, filepath in files[idx].items():
            if not os.path.isfile(filepath):
                raise IOException('File does not exist: %s' % filepath)


def aggregate_paths(pp_dir):
    filenames_dict = {'pet_img': 'nifti_PET.nii',
                     'ct_img': 'nifti_CT.nii',
                     'mask_img': 'nifti_MASK.nii'}
    files = dict()
    subsets = [el for el in ['train', 'val', 'test'] if el in os.listdir(os.path.join(pp_dir))]
    for subset in subsets:
        files[subset] = OrderedDict()
        for study_uid in os.listdir(os.path.join(pp_dir, subset)):
            d = dict()
            for key, filename in filenames_dict.items():
                d[key] = os.path.join(pp_dir, subset, study_uid, filename)

            files[subset][study_uid] = d

    for subset in files:
        check_path_is_valid(files[subset])
    return files


def get_transform(subset, modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin = None,  data_augmentation=True, from_pp=False, cache_pp=False):
    """[summary]

    Args:
        
        subset ([type]): [description]
        modalities ([tuple]): [('pet_img, ct_img') or ('pet_img')]
        mode ([list]): [binary, probs or mean_props] !!!!!
        method ([list]): [relative, absolute, otsu, otsu_abs] !!!!!! CHECK HERE 
        tval ([dict]) : [if mode = binary & method = relative : 0.42
                          if mode = binary & method = absolute : 2.5, 
                          else : don't need tval, tval = 0.0 ]

    """
    
    keys = tuple(list(modalities) + ['mask_img'] + ['merged_img'])
    transformers = [LoadNifti(keys=keys)]  # Load NIFTI file from path

    transformers.append(ResampleMask(keys=('merged_img', 'mask_img'), target_size=target_size, target_spacing=target_spacing, target_direction=target_direction, target_origin=target_origin))
    if not from_pp:

        # Generate ground-truth from PET and ROIs
        if mode == 'binary':
            transformers.append(Roi2Mask(keys=('merged_img', 'mask_img'),
                                         method=method, tval=tval))
        else : 
            transformers.append(
                Roi2MaskProbs(keys=('merged_img', 'mask_img'), mode=mode, method=method,
                              new_key_name='mask_img'))

    if cache_pp:
        transformers = Compose(transformers)
        return transformers

    #Dissociated PET and CT 
    transformers.append(DissociatePETCT(keys=('merged_img'), new_key_name=('pet_img', 'ct_img')))


    # Add Data augmentation
    if subset == 'train' and data_augmentation:
        translation = 10
        scaling = 0.1
        rotation = (np.pi / 60, np.pi / 30, np.pi / 60)
        transformers.append(RandAffine(keys=('pet_img', 'ct_img', 'mask_img'), translation=translation, scaling=scaling, rotation=rotation))
    
    # Convert Simple ITK image into numpy 3d-array
    transformers.append(Sitk2Numpy(keys=('pet_img', 'ct_img', 'mask_img')))

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

    transformers.append(AddChannel(keys='mask_img', channel_first=False))
    transformers = Compose(transformers)
    return transformers


def get_transform_test(modalities):
    """transformers for test set 

    Args:
        modalities ([tuple]): [('pet_img, ct_img') or ('pet_img')]

    Returns:
        [type]: [description]
    """
    keys = tuple(list(modalities) + ['merged_img'])
    transformers = [LoadNifti(keys=keys)]  # Load NIFTI file from path
    #transformers.append(ResampleMask(keys=('merged_img', 'mask_img'), target_size=target_size, target_spacing=target_spacing, target_direction=target_direction, target_origin=target_origin))


    transformers.append(DissociatePETCT(keys=('merged_img'), new_key_name=('pet_img', 'ct_img')))
    transformers.append(Sitk2Numpy(keys=keys))

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
def get_data(pp_dir, csv_path, modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin=None , data_augmentation=True, from_pp=False, cache_pp=False, pp_flag=''):
    """Save or not the pre processed data at nifti format

    Returns dataset, and list of transformers for train and val set. 
    """

    if pp_dir is None:

        # Get Data
        DM = DataManager(csv_path=csv_path)
        train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)
        dataset = dict()
        dataset['train'], dataset['val'], dataset['test'] = train_images_paths, val_images_paths, test_images_paths

        # Define generator
        train_transforms = get_transform('train', modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin , data_augmentation = True, from_pp=False, cache_pp=False)
        val_transforms = get_transform('val', modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin ,  data_augmentation = False, from_pp=False, cache_pp=False)

        return dataset, train_transforms, val_transforms

    elif pp_flag == 'done':
        print('Loading from {} ...'.format(pp_dir))
        dataset = aggregate_paths(pp_dir)
        for subset in dataset:
            print('{} in {} set'.format(len(dataset[subset]), subset))

        # Define generator
        train_transforms = get_transform('train', modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin , data_augmentation = True, from_pp=True, cache_pp=False)
        val_transforms = get_transform('val', modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin ,  data_augmentation = False, from_pp=True, cache_pp=False)

        return dataset, train_transforms, val_transforms

    else:
        # remove old files
        if os.path.exists(pp_dir) and os.path.isdir(pp_dir):
            print('cleaning directory : {}'.format(pp_dir))
            shutil.rmtree(pp_dir)

        # copy the config file in the root of the dir
        #if not os.path.exists(pp_dir):
        #    os.makedirs(pp_dir)
        #copyfile(cfg['cfg_path'], os.path.join(pp_dir, 'config.py'))

        # Get Data Paths
        
        DM = DataManager(csv_path=csv_path)
        train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)
        dataset = dict()
        dataset['train'], dataset['val'], dataset['test'] = train_images_paths, val_images_paths, test_images_paths

        # Get preprocessing transformer
        pp_transforms = get_transform('train', modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin , data_augmentation = True, from_pp=False, cache_pp=True)

        print('Preprocessing and saving data at {}'.format(pp_dir))

        start_time = time.time()
        total_count = -1
        total = np.sum([len(data) for subset, data in dataset.items()])
        for subset, data in dataset.items():
            #
            print('{} : {} examples'.format(subset, len(data)))
            # for count, img_path in enumerate(tqdm(data)):
            for count, img_path in enumerate(data):
                total_count += 1
                current_time = int(time.time() - start_time)

                if total_count == 0:
                    print('[{:>3}/{:>3}]: total_time {}'.format(count, len(data), sec2str(current_time)))
                else:
                    print('[{:>3}/{:>3}]: total_time {}, mean loop {}, ETA {}'.format(
                        count, len(data),
                        sec2str(current_time),
                        sec2str(int(current_time / total_count)),
                        sec2str(int((total - total_count) * current_time / total_count)))
                        )

                result_dict = pp_transforms(img_path)
                study_uid = result_dict['image_id']

                folder_path = os.path.join(pp_dir, subset, study_uid)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # save PET, CT as NIFTI
                pp_filenames_dict =  {'pet_img': 'nifti_PET.nii',
                     'ct_img': 'nifti_CT.nii',
                     'mask_img': 'nifti_MASK.nii'}
                for modality in modalities:
                    sitk.WriteImage(result_dict[modality],
                                    os.path.join(folder_path, pp_filenames_dict[modality]))
                # save MASK as NIFTI
                sitk.WriteImage(result_dict['mask_img'],
                                os.path.join(folder_path, pp_filenames_dict['mask_img']))
                # sitk.WriteImage(result_dict['pet_img'], os.path.join(folder_path, 'nifti_PET.nii'))
                # sitk.WriteImage(result_dict['ct_img'], os.path.join(folder_path, 'nifti_CT.nii'))
                # sitk.WriteImage(result_dict['mask_img'], os.path.join(folder_path, 'nifti_MASK.nii'))

        # set flag
        pp_flag = 'done'

        print('Loading from {} ...'.format(pp_dir))
        dataset = aggregate_paths(pp_dir)
        for subset in dataset:
            print('{} in {} set'.format(len(dataset[subset]), subset))

        # Define generator
        train_transforms = get_transform('train', modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin , data_augmentation = True, from_pp=True, cache_pp=False)
        val_transforms = get_transform('val', modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin ,  data_augmentation = False, from_pp=True, cache_pp=False)

        return dataset, train_transforms, val_transforms
        

