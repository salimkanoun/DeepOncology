
import argparse

import pandas as pd
import numpy as np
import pickle
import os
import json

from class_modalities.datasets import DataManager

from class_modalities.transforms import LoadNifti, Compose, Roi2Mask_probs, ResampleReshapeAlign, Sitk2Numpy, ScaleIntensityRanged
import SimpleITK as sitk


# def aggregate_meta_info(exp_dir):
#
#     files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
#     df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
#     for f in files:
#         with open(f, 'rb') as handle:
#             df.loc[len(df)] = pickle.load(handle)
#
#     df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
#     print ("aggregated meta info to df with length", len(df))


def main(config, args):
    pp_dir = args.pp_dir
    csv_path = config['path']['csv_path']

    # PET CT scan params
    image_shape = tuple(config['preprocessing']['image_shape'])  # (128, 64, 64)  # (368, 128, 128)  # (z, y, x)
    voxel_spacing = tuple(config['preprocessing']['voxel_spacing'])  # (4.8, 4.8, 4.8)  # in millimeter, (z, y, x)
    data_augment = False
    origin = config['preprocessing']['origin']  # how to set the new origin

    # Get Data
    DM = DataManager(csv_path=csv_path)
    train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)

    target_shape = image_shape[::-1]  # (z, y, x) to (x, y, z)
    target_voxel_spacing = voxel_spacing[::-1]

    transformers2 = Compose([  # read img + meta info
        LoadNifti(keys=("pet_img", "ct_img", "mask_img")),
        Roi2Mask_probs(keys=('pet_img', 'mask_img'),
                       method='absolute', new_key_name='mask_img_absolute'),
        Roi2Mask_probs(keys=('pet_img', 'mask_img'),
                       method='relative', new_key_name='mask_img_relative'),
        Roi2Mask_probs(keys=('pet_img', 'mask_img'),
                       method='otsu', new_key_name='mask_img_otsu'),
        ResampleReshapeAlign(target_shape, target_voxel_spacing,
                             keys=['pet_img', "ct_img",
                                   'mask_img_absolute', 'mask_img_relative', 'mask_img_otsu'],
                             origin='head', origin_key='pet_img',
                             interpolator={'pet_img': sitk.sitkLinear,
                                           'ct_img': sitk.sitkLinear,
                                           'mask_img': sitk.sitkLinear,
                                           'mask_img_absolute': sitk.sitkLinear,
                                           'mask_img_relative': sitk.sitkLinear,
                                           'mask_img_otsu': sitk.sitkLinear},
                             default_value={'pet_img': 0.0,
                                            'ct_img': -1000.0,
                                            'mask_img': 0,
                                            'mask_img_absolute': 0.0,
                                            'mask_img_relative': 0.0,
                                            'mask_img_otsu': 0.0}),
        Sitk2Numpy(keys=['pet_img', 'ct_img',
                         'mask_img_absolute', 'mask_img_relative', 'mask_img_otsu']),
        # normalize input
        ScaleIntensityRanged(keys="pet_img",
                             a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys="ct_img",
                             a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True)
    ])

    for dataset, subset in zip([train_images_paths, val_images_paths, test_images_paths], ['train', 'val', 'test']):
        print(subset, len(dataset))

        # create folder
        if not os.path.exists(os.path.join(pp_dir, subset)):
            os.makedirs(os.path.join(pp_dir, subset))

        for idx, img_path in enumerate(dataset):
            result = transformers2(img_path)
            study_uid = result['image_id']

            # create folder
            base_path = os.path.join(pp_dir, subset, study_uid)
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            for key in ['mask_img_absolute', 'mask_img_relative', 'mask_img_otsu', 'pet_img', 'ct_img']:
                np.save(os.path.join(base_path, key), result[key])

            with open(os.path.join(base_path, 'meta_info.pickle'), 'wb') as handle:
                meta_info_dict = {'study_uid': study_uid,
                                  'axes': 'zyx', 'shape': image_shape, 'spacing': voxel_spacing}
                volume_voxel = np.prod(voxel_spacing) * 10**-6  # volume of one voxel in liters
                # add tmtv
                for key in ['mask_img_absolute', 'mask_img_relative', 'mask_img_otsu']:
                    meta_info_dict['tmtv_' + key] = np.sum(np.round(result[key])) * volume_voxel

                mean_mask = np.mean([result[key] for key in ['mask_img_absolute', 'mask_img_relative', 'mask_img_otsu']],
                                    axis=0)
                meta_info_dict['tmtv_mean_mask'] = np.sum(np.round(mean_mask)) * volume_voxel

                pickle.dump(meta_info_dict, handle)

            print('[{} / {}] : Succesfully saved {}'.format(idx, len(dataset), study_uid))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/default_config.json', type=str,
                        help="path/to/config.json")
    parser.add_argument("--pp_dir", type=str, required=True,
                        help="path/to/preprocessing/directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    main(config, args)



