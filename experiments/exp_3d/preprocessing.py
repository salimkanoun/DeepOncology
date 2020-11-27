
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


def check_path_is_valid(files):
    indexes = files.keys() if isinstance(files, dict) else np.arange(len(files))

    for idx in indexes:
        for key, filepath in files[idx].items():
            if not os.path.isfile(filepath):
                raise IOException('File does not exist: %s' % filepath)


def aggregate_paths(cfg):
    pp_dir = cfg['pp_dir']

    filenames_dict = cfg['pp_filenames_dict']

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


def get_transform(cfg, subset, from_pp=False, cache_pp=False):
    keys = tuple(list(cfg['modalities']) + ['mask_img'])
    transformers = [LoadNifti(keys=keys)]  # Load NIFTI file from path

    if not from_pp:

        # Generate ground-truth from PET and VOIs
        if cfg['mode'] == 'binary':
            transformers.append(Roi2Mask(keys=('pet_img', 'mask_img'),
                                         method=cfg['method'], tval=cfg['tvals_binary'].get(cfg['method'], 0.0)))
        elif cfg['mode'] == 'probs' and cfg['method'] == 'otsu_abs':
            transformers.append(Roi2MaskOtsuAbsolute(keys=('pet_img', 'mask_img'), tvals_probs=cfg['tvals_probs'],
                                                     new_key_name='mask_img'))
        elif cfg['mode'] == 'probs' or cfg['mode'] == 'mean_probs':
            transformers.append(
                Roi2MaskProbs(keys=('pet_img', 'mask_img'), method=cfg['method'], tvals_probs=cfg['tvals_probs'],
                              new_key_name='mask_img'))

        # Resample, reshape and align to the same view
        transformers.append(ResampleReshapeAlign(keys=keys,
                                                 target_shape=cfg['image_shape'][::-1],
                                                 target_voxel_spacing=cfg['voxel_spacing'][::-1],
                                                 origin=cfg['origin'], origin_key='pet_img',
                                                 interpolator=cfg['pp_kwargs']['interpolator'],
                                                 default_value=cfg['pp_kwargs']['default_value']))
    if cache_pp:
        transformers = Compose(transformers)
        return transformers

    # Add Data augmentation
    if subset == 'train' and cfg['data_augmentation']:
        transformers.append(RandAffine(keys=keys,
                                       **cfg['da_kwargs']))
    # Convert Simple ITK image into numpy 3d-array
    transformers.append(Sitk2Numpy(keys=keys))

    # Normalize input values
    for modality in cfg['modalities']:
        transformers.append(ScaleIntensityRanged(keys=modality,
                                                 **cfg['pp_kwargs'][modality]))
    # Concatenate modalities if necessary
    if len(cfg['modalities']) > 1:
        transformers.append(ConcatModality(keys=cfg['modalities'], channel_first=False, new_key='input'))
    else:
        transformers.append(AddChannel(keys=cfg['modalities'], channel_first=False))
        transformers.append(RenameDict(keys=cfg['modalities'], keys2='input'))

    transformers.append(AddChannel(keys='mask_img', channel_first=False))
    transformers = Compose(transformers)
    return transformers


def get_transform_test(cfg, from_pp=False):
    keys = cfg['modalities']
    transformers = [LoadNifti(keys=keys)]  # Load NIFTI file from path

    if not from_pp:
        # Resample, reshape and align to the same view
        transformers.append(ResampleReshapeAlign(keys=keys,
                                                 target_shape=cfg['image_shape'][::-1],
                                                 target_voxel_spacing=cfg['voxel_spacing'][::-1],
                                                 origin=cfg['origin'], origin_key='pet_img',
                                                 interpolator=cfg['pp_kwargs']['interpolator'],
                                                 default_value=cfg['pp_kwargs']['default_value']))
    transformers.append(Sitk2Numpy(keys=keys))

    # Normalize input values
    for modality in cfg['modalities']:
        transformers.append(ScaleIntensityRanged(keys=modality,
                                                 **cfg['pp_kwargs'][modality]))
    # Concatenate modalities if necessary
    if len(cfg['modalities']) > 1:
        transformers.append(ConcatModality(keys=cfg['modalities'], channel_first=False, new_key='input'))
    else:
        transformers.append(AddChannel(keys=cfg['modalities'], channel_first=False))
        transformers.append(RenameDict(keys=cfg['modalities'], keys2='input'))

    transformers = Compose(transformers)
    return transformers


def get_data(cfg):
    pp_dir = cfg.get('pp_dir', None)

    if pp_dir is None:
        csv_path = cfg['csv_path']

        # Get Data
        DM = DataManager(csv_path=csv_path)
        train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)
        dataset = dict()
        dataset['train'], dataset['val'], dataset['test'] = train_images_paths, val_images_paths, test_images_paths

        # Define generator
        train_transforms = get_transform(cfg, 'train', from_pp=False)
        val_transforms = get_transform(cfg, 'val', from_pp=False)

        return dataset, train_transforms, val_transforms

    elif cfg.get('pp_flag', '') == 'done':
        print('Loading from {} ...'.format(pp_dir))
        dataset = aggregate_paths(cfg)
        for subset in dataset:
            print('{} in {} set'.format(len(dataset[subset]), subset))

        # Define generator
        train_transforms = get_transform(cfg, 'train', from_pp=True)
        val_transforms = get_transform(cfg, 'val', from_pp=True)

        return dataset, train_transforms, val_transforms

    else:
        # remove old files
        if os.path.exists(pp_dir) and os.path.isdir(pp_dir):
            print('cleaning directory : {}'.format(pp_dir))
            shutil.rmtree(pp_dir)

        # copy the config file in the root of the dir
        if not os.path.exists(pp_dir):
            os.makedirs(pp_dir)
        copyfile(cfg['cfg_path'], os.path.join(pp_dir, 'config.py'))

        # Get Data Paths
        csv_path = cfg['csv_path']
        DM = DataManager(csv_path=csv_path)
        train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)
        train_images_paths, val_images_paths, test_images_paths = train_images_paths[:50], val_images_paths[:20], test_images_paths[:10]
        dataset = dict()
        dataset['train'], dataset['val'], dataset['test'] = train_images_paths, val_images_paths, test_images_paths

        # Get preprocessing transformer
        pp_transforms = get_transform(cfg, 'train', from_pp=False, cache_pp=True)

        print('Preprocessing and saving data at {}'.format(pp_dir))

        start_time = time.time()
        total_count = -1
        total = np.sum([len(data) for subset, data in dataset.items()])
        for subset, data in dataset.items():
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
                for modality in cfg['modalities']:
                    sitk.WriteImage(result_dict[modality],
                                    os.path.join(folder_path, cfg['pp_filenames_dict'][modality]))
                # save MASK as NIFTI
                sitk.WriteImage(result_dict['mask_img'],
                                os.path.join(folder_path, cfg['pp_filenames_dict']['mask_img']))
                # sitk.WriteImage(result_dict['pet_img'], os.path.join(folder_path, 'nifti_PET.nii'))
                # sitk.WriteImage(result_dict['ct_img'], os.path.join(folder_path, 'nifti_CT.nii'))
                # sitk.WriteImage(result_dict['mask_img'], os.path.join(folder_path, 'nifti_MASK.nii'))

        # set flag
        cfg['pp_flag'] = 'done'

        return get_data(cfg)

