import argparse
import json

import os
from datetime import datetime

# from lib.transforms import *
from experiments.exp_3d.preprocessing import *

import tensorflow as tf

from lib.utils import read_cfg
import csv
from tqdm import tqdm
import numpy as np
from losses.Metrics import metric_dice


def main(cfg, args):
    # path
    now = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    # Get Data path and transforms
    data, train_transforms, val_transforms = get_data(cfg)

    model_path = 'model_weights.h5' if args.weight == '' else args.weight
    model_cnn = tf.keras.models.load_model(model_path, compile=False)

    x_key = 'input'
    y_key = 'mask_img'

    result_csv_path = os.path.join(args.target_dir, 'result_tmtv.csv')
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    if not os.path.isfile(result_csv_path):
        # csv file do not exist yet => so let's create it
        with open(result_csv_path, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['study_uid', 'subset', 'model', 'config',
                             'dice_cnn', 'tmtv_cnn_pred', 'tmtv_cnn_true'])

    with open(result_csv_path, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        for subset, dataset in data.items():
            print('dataset {} : {}'.format(subset, len(dataset)))

            for idx, img_dict in enumerate(tqdm(dataset)):
                # preprocessing
                img_dict = val_transforms(img_dict)
                study_uid = img_dict['image_id']
                img = img_dict[x_key]
                gt = img_dict[y_key]
                gt = np.round(gt)

                # predict cnn
                img = np.expand_dims(img, axis=0)
                pred = model_cnn.predict(img)
                pred = np.round(pred)

                # compute metric and tmtv
                gt = np.squeeze(gt)
                pred = np.squeeze(pred)
                voxel_spacing = img_dict['meta_info']['new_spacing']
                volume_voxel = np.prod(voxel_spacing) * 10 ** -6  # volume of one voxel in liter
                dice = metric_dice(gt, pred, axis=(0, 1, 2))
                tmtv_true = np.sum(gt, axis=(0, 1, 2)) * volume_voxel
                tmtv_pred = np.sum(pred, axis=(0, 1, 2)) * volume_voxel

                # save result
                writer.writerow(
                    [study_uid, subset, args.model_path, config['cfg_path'], dice, tmtv_pred, tmtv_true])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/config_3d.py', type=str,
                        help="path/to/config.py")
    parser.add_argument('-w', "--weight", default='', type=str,
                        help='path/to/model/weight.h5')
    parser.add_argument("-t", "--target_dir", type=str,
                        help="path/to/target/directory i.e where the pred and csv will is saved")
    args = parser.parse_args()

    config = read_cfg(args.config)
    config['cfg_path'] = args.config

    main(config, args)
