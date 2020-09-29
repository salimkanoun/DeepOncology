import argparse
import json
import csv
from tqdm import tqdm

import os
import numpy as np

from class_modalities.datasets import DataManager
from class_modalities.modality_PETCT import DataGenerator
from class_modalities.data_loader import DataGenerator_3D_from_numpy

from deeplearning_tools.Metrics import metric_dice

import tensorflow as tf


def main(config, args):
    # path
    csv_path = config['path']['csv_path']
    pp_dir = config['path'].get('pp_dir', None)

    # PET CT scan params
    image_shape = tuple(config['preprocessing']['image_shape'])  # (128, 64, 64)  # (368, 128, 128)  # (z, y, x)
    voxel_spacing = tuple(config['preprocessing']['voxel_spacing'])  # (4.8, 4.8, 4.8)  # in millimeter, (z, y, x)
    data_augment = False
    resize = config['preprocessing']['resize']  # True  # not use yet
    origin = config['preprocessing']['origin']  # how to set the new origin
    normalize = config['preprocessing']['normalize']  # True  # whether or not to normalize the inputs

    # Training params
    batch_size = 1
    shuffle = False

    # Get Data
    if pp_dir is None:
        DM = DataManager(csv_path=csv_path)
        x_train, x_val, x_test, y_train, y_val, y_test = DM.get_train_val_test()

        # Define generator
        train_generator = DataGenerator(x_train, y_train,
                                        batch_size=batch_size, shuffle=shuffle, augmentation=data_augment,
                                        target_shape=image_shape, target_voxel_spacing=voxel_spacing,
                                        resize=resize, normalize=normalize, origin=origin)

        val_generator = DataGenerator(x_val, y_val,
                                      batch_size=batch_size, shuffle=False, augmentation=False,
                                      target_shape=image_shape, target_voxel_spacing=voxel_spacing,
                                      resize=resize, normalize=normalize, origin=origin)

        test_generator = DataGenerator(x_test, y_test,
                                       batch_size=batch_size, shuffle=False, augmentation=False,
                                       target_shape=image_shape, target_voxel_spacing=voxel_spacing,
                                       resize=resize, normalize=normalize, origin=origin)
    else:
        mask_keys = ['mask_img_absolute', 'mask_img_relative', 'mask_img_otsu']
        train_generator = DataGenerator_3D_from_numpy(pp_dir, 'train',
                                                      mask_keys,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      returns_dict=True)
        val_generator = DataGenerator_3D_from_numpy(pp_dir, 'val',
                                                    mask_keys,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    returns_dict=True)

        test_generator = DataGenerator_3D_from_numpy(pp_dir, 'test',
                                                     mask_keys,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     returns_dict=True)

    # load model
    model_cnn = tf.keras.models.load_model(args.model_path, compile=False)

    result_csv_path = os.path.join(args.target_dir, 'result_tmtv.csv')
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    if not os.path.isfile(result_csv_path):
        # csv file do not exist yet => so let's create it
        with open(result_csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['study_uid', 'model', 'ground_truth', 'dice_cnn', 'tmtv_cnn_pred', 'tmtv_cnn_true'])

    with open(result_csv_path, 'a') as f:
        writer = csv.writer(f)
        for subset, dataset in zip(['train', 'val', 'test'], [train_generator, val_generator, test_generator]):
            print('dataset {} : {}'.format(subset, len(dataset) * batch_size))

            for idx, img_dict in enumerate(tqdm(dataset)):
                study_uid_batch = img_dict['study_uid']
                X_batch = img_dict['img']
                Y_batch = img_dict['seg']
                Y_batch = np.round(Y_batch)

                Y_pred = model_cnn.predict(X_batch)
                Y_pred = np.round(Y_pred)

                # compute metric and tmtv
                volume_voxel = np.prod(voxel_spacing) * 10 ** -6  # volume of one voxel in liter
                dice_batch = metric_dice(Y_batch, Y_pred, axis=(1, 2, 3, 4))
                tmtv_true = np.sum(Y_batch, axis=(1, 2, 3, 4)) * volume_voxel
                tmtv_pred = np.sum(Y_pred, axis=(1, 2, 3, 4)) * volume_voxel

                for ii in range(X_batch.shape[0]):
                    if args.save_pred:
                        np.save(args.target_dir, np.squeeze(Y_pred[ii]))
                    writer.writerow(
                        [study_uid_batch[ii], args.model_path, 'mean' + '|'.join(mask_keys),
                         dice_batch[ii], tmtv_pred[ii], tmtv_true[ii]])
                    # header =
                    # writer.writerow(
                    #     ['study_uid', 'model', 'ground_truth', 'dice_cnn', 'tmtv_cnn_pred', 'tmtv_cnn_true'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/default_config.json', type=str,
                        help="path/to/config.json")
    parser.add_argument("--model_path", type=str,
                        help="path/to/model.h5")
    parser.add_argument("--target_dir", type=str,
                        help="path/to/target/directory i.e where the pred and csv will is saved")
    parser.add_argument("--save_pred", action='store_true', default=False,
                        help='wheter to save the prediction of the model')
    args = parser.parse_args()

    if args.model_path is None:
        raise argparse.ArgumentError(parser,
                                     "model path was noy supplied.\n" + parser.format_help())

    with open(args.config) as f:
        config = json.load(f)

    main(config, args)
