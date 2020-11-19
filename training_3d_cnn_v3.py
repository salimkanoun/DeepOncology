import sys
import argparse
import json
from shutil import copyfile

from class_modalities.datasets import DataManager
from class_modalities.modality_PETCT import DataGenerator
from class_modalities.data_loader import DataGenerator_3D_from_nifti
from class_modalities.transforms import *

import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from networks.Unet import CustomUNet3D
from networks.Vnet import VNet
from networks.Layers import prelu

from losses.Loss_3d import vnet_dice_loss, custom_robust_loss
from losses.Loss_3d import metric_dice as dsc

import os
from datetime import datetime


def main(config):
    # path
    now = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    # path
    csv_path = config['csv_path']
    pp_dir = config.get('pp_dir', None)

    trained_model_path = config['trained_model_path']  # if None, trained from scratch
    training_model_folder = os.path.join(config['training_model_folder'], now)  # '/path/to/folder'
    if not os.path.exists(training_model_folder):
        os.makedirs(training_model_folder)
    logdir = os.path.join(training_model_folder, 'logs')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # saving the config in the result folder
    copyfile(config['cfg_path'], os.path.join(training_model_folder, 'config.py'))

    # PET CT scan params
    image_shape = config['image_shape']
    number_channels = config['number_channels']
    voxel_spacing = config['voxel_spacing']
    origin = config['origin']  # how to set the new origin
    number_class = config['number_class']  # 2

    # CNN params
    architecture = config['model']['architecture']  # 'unet' or 'vnet'
    cnn_params = config['cnn_params']

    # Training params
    epochs = config['epochs']
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    data_augment = config['data_augment']  # True  # for training dataset only

    # multi gpu training strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # definition of loss, optimizer and metrics
        loss_object = custom_robust_loss
        optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
        metrics = [dsc]

    # callbacks
    callbacks = []
    if config.get('ReduceLROnPlateau', False):
        # reduces learning rate if no improvement are seen
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                    patience=config['training']['callbacks']['patience'],
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.0000001)
        callbacks.append(learning_rate_reduction)

    if config.get('EarlyStopping', False):
        # stop training if no improvements are seen
        early_stop = EarlyStopping(monitor="val_loss",
                                   mode="min",
                                   patience=int(config['training']['callbacks']['patience'] // 2),
                                   restore_best_weights=True)
        callbacks.append(early_stop)

    if config.get('ModelCheckpoint', False):
        # saves model weights to file
        # 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpoint = ModelCheckpoint(os.path.join(training_model_folder, 'model_weights.h5'),
                                     monitor='val_loss',  # metric_dice
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',  # max
                                     save_weights_only=False)
        callbacks.append(checkpoint)

    if config.get('TensorBoard', False):
        tensorboard_callback = TensorBoard(log_dir=logdir,
                                           histogram_freq=0,
                                           batch_size=batch_size,
                                           write_graph=True,
                                           write_grads=True,
                                           write_images=False)
        callbacks.append(tensorboard_callback)

    dataset, train_transforms, val_transforms = get_data(config)
    train_images_paths = dataset['train']
    val_images_paths = dataset['val']

    train_generator = DataGenerator_3D_from_nifti(train_images_paths,
                                                  train_transforms,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  x_key='input', y_key='output')

    val_generator = DataGenerator_3D_from_nifti(val_images_paths,
                                                val_transforms,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                x_key='input', y_key='output')

    # Define model
    if architecture == 'unet':
        with strategy.scope():
            model = CustomUNet3D(tuple(list(image_shape) + [number_channels]), number_class,
                                 **cnn_params).create_model()

    elif architecture == 'vnet':
        with strategy.scope():
            model = VNet(tuple(list(image_shape) + [number_channels]), number_class, **cnn_params).create_model()
    else:
        raise ValueError('Architecture ' + architecture + ' not supported. Please ' +
                         'choose one of unet|vnet.')
    with strategy.scope():
        model.compile(loss=loss_object, optimizer=optimizer, metrics=metrics)

    if trained_model_path is not None:
        with strategy.scope():
            model.load_weights(trained_model_path)

    print(model.summary())

    # serialize model to JSON before training
    model_json = model.to_json()
    with open(os.path.join(training_model_folder, 'architecture_{}_model_{}.json'.format(architecture, now)),
              "w") as json_file:
        json_file.write(model_json)

    # training model
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        epochs=epochs,
                        callbacks=callbacks,  # initial_epoch=0,
                        verbose=1
                        )

    # whole model saving
    model.save(os.path.join(training_model_folder, 'trained_model_{}.h5'.format(now)))


def get_data(cfg):

    # Get Data
    DM = DataManager(csv_path=cfg.csv_path)
    train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)

    dataset = dict()
    dataset['train'] = train_images_paths
    dataset['val'] = val_images_paths
    dataset['test'] = test_images_paths

    # Define generator
    train_transforms = Compose([LoadNifti(keys=("pet_img", "ct_img", "mask_img")),
                                Roi2Mask_otsu_absolute(keys=('pet_img', 'mask_img'), new_key_name='output'),
                                ResampleReshapeAlign(target_shape=cfg.image_shape[::-1],
                                                     target_voxel_spacing=cfg.voxel_spacing[::-1],
                                                     keys=['pet_img', "ct_img", 'output'],
                                                     origin='head', origin_key='pet_img',
                                                     interpolator={'pet_img': sitk.sitkLinear,
                                                                   'ct_img': sitk.sitkLinear,
                                                                   'output': sitk.sitkLinear},
                                                     default_value={'pet_img': 0.0,
                                                                    'ct_img': -1000.0,
                                                                    'output': 0.0}),
                                RandAffine(keys=['pet_img', "ct_img", 'output'],
                                           translation=10, scaling=0.1, rotation=(pi / 60, pi / 30, pi / 60),
                                           interpolator={'pet_img': sitk.sitkLinear,
                                                         'ct_img': sitk.sitkLinear,
                                                         'output': sitk.sitkLinear},
                                           default_value={'pet_img': 0.0,
                                                          'ct_img': -1000.0,
                                                          'output': 0.0}),
                                Sitk2Numpy(keys=['pet_img', 'ct_img', 'output']),
                                # normalize input
                                ScaleIntensityRanged(keys="pet_img",
                                                     a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True),
                                ScaleIntensityRanged(keys="ct_img",
                                                     a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
                                ConcatModality(keys=('pet_img', 'ct_img'), channel_first=False, new_key='input'),
                                ])

    val_transforms = Compose([LoadNifti(keys=("pet_img", "ct_img", "mask_img")),
                              Roi2Mask_otsu_absolute(keys=('pet_img', 'mask_img'), new_key_name='output'),
                              ResampleReshapeAlign(target_shape=image_shape[::-1],
                                                   target_voxel_spacing=voxel_spacing[::-1],
                                                   keys=['pet_img', "ct_img", 'output'],
                                                   origin='head', origin_key='pet_img',
                                                   interpolator={'pet_img': sitk.sitkLinear,
                                                                 'ct_img': sitk.sitkLinear,
                                                                 'output': sitk.sitkLinear},
                                                   default_value={'pet_img': 0.0,
                                                                  'ct_img': -1000.0,
                                                                  'output': 0.0}),
                              Sitk2Numpy(keys=['pet_img', 'ct_img', 'output']),
                              # normalize input
                              ScaleIntensityRanged(keys="pet_img",
                                                   a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True),
                              ScaleIntensityRanged(keys="ct_img",
                                                   a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
                              ConcatModality(keys=('pet_img', 'ct_img'), channel_first=False, new_key='input'),
                              ])

    return dataset, train_transforms, val_transforms


def read_cfg(filepath):
    var = dict()
    exec(open(filepath).read(), var)

    return var


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/default_config.json', type=str,
                        help="path/to/config.json")
    args = parser.parse_args()

    config = read_cfg(args.config)
    config['cfg_path'] = args.config

    main(config)
