import sys
import argparse
import json

from class_modalities.datasets import DataManager
from class_modalities.modality_PETCT import DataGenerator

import numpy as np
import collections
import re

import os
from datetime import datetime


def main(config, args):
    # path
    csv_path = config['path']['csv_path']

    # PET CT scan params
    image_shape = tuple(config['preprocessing']['image_shape'])  # (128, 64, 64)  # (368, 128, 128)  # (z, y, x)
    number_channels = config['preprocessing']['number_channels']
    voxel_spacing = tuple(config['preprocessing']['voxel_spacing'])  # (4.8, 4.8, 4.8)  # in millimeter, (z, y, x)
    data_augment = False
    origin = config['preprocessing']['origin']  # how to set the new origin
    normalize = config['preprocessing']['normalize']  # True  # whether or not to normalize the inputs
    number_class = config['preprocessing']['number_class']  # 2

    # Get Data
    DM = DataManager(csv_path=csv_path)
    train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/default_config.json', type=str,
                        help="path/to/config.json")
    parser.add_argument("--pp_dir", type=str,
                        help="path/to/preprocessing/directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    main(config, args)

# for subset_type in dataset.keys():
for subset_type in ['train', 'val', 'test']:
    print(subset_type)
    # path to pdf to generate
    folder_name = os.path.join(data_path, subset_type)
    print('folder :', folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(folder_name, 'folder created')

    # Define generator
    generator = DataGenerator(dataset[subset_type]['x'], dataset[subset_type]['y'],
                              batch_size=1, shuffle=False, augmentation=False,
                              target_shape=image_shape, target_voxel_spacing=voxel_spacing,
                              resize=resize, normalize=normalize)

    # loop on files to get MIP visualisation
    for step, (X, mask) in enumerate(generator):
        pet_path = dataset['subset_type']['x'][step][0]
        study_uid = re.sub('_nifti_PT\.nii(\.gz)?', '', os.path.basename(pet_path))

        pet_array = X[0, :, :, :, 0]
        ct_array = X[0, :, :, :, 1]
        mask_array = mask[0]

        if not os.path.exists(os.path.join(folder_name, study_uid)):
            os.makedirs(os.path.join(folder_name, study_uid))

        np.save(os.path.join(folder_name, study_uid, study_uid + '_PET.npy'), pet_array)
        np.save(os.path.join(folder_name, study_uid, study_uid + '_CT.npy'), ct_array)
        np.save(os.path.join(folder_name, study_uid, study_uid + '_MASK.npy'), mask_array)

        print('Succesfully saved :', step, study_uid)


