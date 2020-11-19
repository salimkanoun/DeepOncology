
import sys
import argparse
import json
import csv

import numpy as np
import SimpleITK as sitk

from class_modalities.datasets import DataManager
from class_modalities.modality_PETCT_2D import FullPipeline
from class_modalities.utils import get_study_uid
from losses.Metrics import metric_dice

import os
from datetime import datetime


def main(config):

    # path
    csv_path = config['path']['csv_path']
    model_path = "/media/salim/DD 2To/RUDY_WEIGTH/training/20200827-08:18:39/model_weights.h5"

    # PET CT scan params
    image_shape = tuple(config['preprocessing']['image_shape'])  # (z, y, x)
    in_channels, out_channels = config['model']['in_channels'], config['model']['out_channels']  # 6, 1
    voxel_spacing = tuple(config['preprocessing']['voxel_spacing'])  # in millimeter, (z, y, x)
    normalize = config['preprocessing']['normalize']  # True  # whether or not to normalize the inputs

    # Get Pipeline
    pipeline = FullPipeline(model_path,
                            target_shape=image_shape,
                            target_voxel_spacing=voxel_spacing,
                            normalize=True,
                            in_channels=6,
                            threshold='otsu')

    # Get Data
    DM = DataManager(csv_path=csv_path)
    train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)

    for subset, dataset in zip(['train', 'val', 'test'], [train_images_paths, val_images_paths, test_images_paths]):
        print(subset)
        header = ['study_uid', 'subset', 'metric_dice_cnn', 'metric_dice_pipeline',
                  'tmtv_cnn_pred', 'tmtv_cnn_true', 'tmtv_pipeline_pred', 'tmtv_pipeline_true']

        with open('/home/salim/Documents/DeepOncopole/data/densexnet/result_tmtv_{}.csv'.format(subset), 'w') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(header)

            step = 0
            for img_path in dataset:
                step += 1
                study_uid = get_study_uid(img_path['pet_img'])
                print('[%4d / %4d] : %s' % (step, len(dataset), study_uid))

                # predict cnn + resample etc
                result = pipeline(img_path)

                # NIfTI to numpy
                mask_cnn_true = sitk.GetArrayFromImage(result['mask_cnn_true'])
                mask_cnn_pred = np.round(sitk.GetArrayFromImage(result['mask_cnn_pred']))

                mask_true = sitk.GetArrayFromImage(result['mask_true'])
                mask_pred = sitk.GetArrayFromImage(result['mask_pred'])

                # metric
                metrice_dice_cnn = metric_dice(mask_cnn_true, mask_cnn_pred)
                metrice_dice_pipeline = metric_dice(mask_true, mask_pred)

                # tmtv directly with the output of the cnn
                spacing = result['mask_cnn_true'].GetSpacing()
                unit_vol_cnn = spacing[0] * spacing[1] * spacing[2]
                tmtv_cnn_pred = np.sum(mask_cnn_pred) * unit_vol_cnn
                tmtv_cnn_true = np.sum(mask_cnn_true) * unit_vol_cnn

                # tmtv for entire pipeline
                spacing = result['mask_true'].GetSpacing()
                unit_vol = spacing[0] * spacing[1] * spacing[2]
                tmtv_pipeline_pred = np.sum(mask_pred) * unit_vol
                tmtv_pipeline_true = np.sum(mask_true) * unit_vol

                # write the result
                writer.writerow([study_uid, subset, metrice_dice_cnn, metrice_dice_pipeline,
                                 tmtv_cnn_pred, tmtv_cnn_true, tmtv_pipeline_pred, tmtv_pipeline_true])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/default_config_2d.json', type=str,
                        help="path/to/config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    main(config)
