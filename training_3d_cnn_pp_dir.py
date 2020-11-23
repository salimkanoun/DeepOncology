import argparse
import json

from class_modalities.datasets import DataManager
from class_modalities.data_loader import DataGeneratorFromDict
from class_modalities.transforms import *

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from deeplearning_models.Unet import CustomUNet3D
from deeplearning_models.Vnet import VNet
from deeplearning_models.Layers import prelu

from deeplearning_tools.Loss import vnet_dice_loss, custom_robust_loss, rce_loss, w_rce_loss
from deeplearning_tools.Loss import metric_dice as dsc
from deeplearning_tools.Loss import metric_dice_sigmoid as dsc_sigmoid

import os
from datetime import datetime

from collections import OrderedDict


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
    for subset in os.listdir(os.path.join(pp_dir)):
        files[subset] = OrderedDict()
        for study_uid in os.listdir(os.path.join(pp_dir, subset)):
            d = dict()
            for key, filename in filenames_dict.items():
                d[key] = os.path.join(pp_dir, subset, study_uid, filename)

            files[subset][study_uid] = d

    for subset in files:
        check_path_is_valid(files[subset])
    return files


def get_transform(cfg, subset, from_pp=False):
    keys = tuple(list(cfg.modalities) + ['mask_img'])
    transformers = [LoadNifti(keys=keys)]

    if cfg.mode == 'binary':
        transformers.append(Roi2Mask(keys=('pet_img', 'mask_img'),
                                     method=cfg.method, tval=cfg.tvals_binary.get(method, 0.0)))
    elif cfg.mode == 'probs' and cfg.method == 'otsu_abs':
            transformers.append(Roi2MaskOtsuAbsolute(keys=('pet_img', 'mask_img'), tvals_probs=cfg.tvals_probs,
                                                     new_key_name='mask_img'))
    elif cfg.mode == 'probs' or cfg.mode == 'mean_probs':
        transformers.append(Roi2MaskProbs(eys=('pet_img', 'mask_img'), method=cfg.method, tvals_probs=cfg.tvals_probs,
                                          new_key_name='mask_img'))

    transformers.append(ResampleReshapeAlign(target_shape=cfg.image_shape[::-1],
                                             target_voxel_spacing=cfg.voxel_spacing[::-1],
                                             keys=keys,
                                             origin=cfg['origin'], origin_key='pet_img',
                                             interpolator=cfg['pp_kwargs']['interpolator'],
                                             default_value=cfg['pp_kwargs']['default_value']))

    if subset == 'train' and cfg.data_augmentation:
        transformers.append(RandAffine(keys=keys,
                                       **cfg.da_kwargs))
    transformers.append(Sitk2Numpy(keys=keys))

    for modality in cfg.modalities:
        transformers.append(ScaleIntensityRanged(keys=modality,
                                                 **cfg.pp_kwargs[modality]))

    transfomers.append(ConcatModality(keys=('pet_img', 'ct_img'), channel_first=False, new_key='input'))

    transformers = Compose(transformers)
    return transformers


def get_data(cfg):
    pp_dir = config['path'].get('pp_dir', None)

    if pp_dir is None:
        csv_path = cfg['csv_path']

        image_shape = tuple(cfg['image_shape'])
        voxel_spacing = tuple(cfg['voxel_spacing'])

        # Get Data
        DM = DataManager(csv_path=csv_path)
        train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)
        dataset = dict()
        dataset['train'], dataset['val'], dataset['test'] = train_images_paths, val_images_paths, test_images_paths

        # Define generator
        train_transforms = Compose([LoadNifti(keys=("pet_img", "ct_img", "mask_img")),
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
                                    ConcatModality(keys=('pet_img', 'ct_img'), channel_first=False, new_key='input')
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
                                  ConcatModality(keys=('pet_img', 'ct_img'), channel_first=False, new_key='input')
                                  ])

        return dataset, train_transforms, val_transforms

    else:
        print('Loading from {} ...'.format(pp_dir))
        dataset = aggregate_paths(pp_dir)
        for subset in dataset:
            print('{} in {} set'.format(len(dataset[subset]), subset))

        # Define generator
        train_transforms = Compose([LoadNifti(keys=("pet_img", "ct_img", "mask_img"),
                                              dtypes={'pet_img': sitk.sitkFloat32,
                                                      'ct_img': sitk.sitkFloat32,
                                                      'mask_img': sitk.sitkFloat32}  # sitk.sitkUInt8}
                                              ),
                                    RandAffine(keys=['pet_img', "ct_img", 'mask_img'],
                                               translation=10, scaling=0.1, rotation=(pi / 60, pi / 30, pi / 60),
                                               interpolator={'pet_img': sitk.sitkLinear,
                                                             'ct_img': sitk.sitkLinear,
                                                             'mask_img': sitk.sitkLinear},
                                               default_value={'pet_img': 0.0,
                                                              'ct_img': -1000.0,
                                                              'mask_img': 0.0}),
                                    Sitk2Numpy(keys=['pet_img', 'ct_img', 'mask_img']),
                                    # normalize input
                                    ScaleIntensityRanged(keys="pet_img",
                                                         a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True),
                                    ScaleIntensityRanged(keys="ct_img",
                                                         a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
                                    ConcatModality(keys=('pet_img', 'ct_img'), channel_first=False, new_key='input')
                                    ])

        val_transforms = Compose([LoadNifti(keys=("pet_img", "ct_img", "mask_img"),
                                            dtypes={'pet_img': sitk.sitkFloat32,
                                                    'ct_img': sitk.sitkFloat32,
                                                    'mask_img': sitk.sitkFloat32}
                                            ),
                                  Sitk2Numpy(keys=['pet_img', 'ct_img', 'mask_img']),
                                  # normalize input
                                  ScaleIntensityRanged(keys="pet_img",
                                                       a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True),
                                  ScaleIntensityRanged(keys="ct_img",
                                                       a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
                                  ConcatModality(keys=('pet_img', 'ct_img'), channel_first=False, new_key='input')
                                  ])

        return dataset, train_transforms, val_transforms


def main(config):
    # path
    now = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    # path
    csv_path = config['path']['csv_path']
    pp_dir = config['path'].get('pp_dir', None)

    trained_model_path = config['path']['trained_model_path']  # if None, trained from scratch
    training_model_folder = os.path.join(config['path']['training_model_folder'], now)  # '/path/to/folder'
    if not os.path.exists(training_model_folder):
        os.makedirs(training_model_folder)
    logdir = os.path.join(training_model_folder, 'logs')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # saving the config in the result folder
    with open(os.path.join(training_model_folder, 'config.json'), 'w') as f:
        json.dump(config, f)

    # PET CT scan params
    image_shape = tuple(config['preprocessing']['image_shape'])  # (128, 64, 64)  # (368, 128, 128)  # (z, y, x)
    number_channels = config['preprocessing']['number_channels']
    voxel_spacing = tuple(config['preprocessing']['voxel_spacing'])  # (4.8, 4.8, 4.8)  # in millimeter, (z, y, x)
    data_augment = config['preprocessing']['data_augment']  # True  # for training dataset only
    origin = config['preprocessing']['origin']  # how to set the new origin
    normalize = config['preprocessing']['normalize']  # True  # whether or not to normalize the inputs
    number_class = config['preprocessing']['number_class']  # 2
    threshold = config["preprocessing"]['threshold']

    # CNN params
    architecture = config['model']['architecture']  # 'unet' or 'vnet'

    cnn_params = config['model'][architecture]['cnn_params']
    # transform list to tuple
    for key, value in cnn_params.items():
        if isinstance(value, list):
            cnn_params[key] = tuple(value)
    # get activation layer from name
    if cnn_params["activation"] == 'prelu':
        cnn_params["activation"] = prelu
    else:
        cnn_params["activation"] = tf.keras.activations.get(cnn_params["activation"])

    # Training params
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    shuffle = config['training']['shuffle']

    # multi gpu training strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # definition of loss, optimizer and metrics
        loss_object = custom_robust_loss
        optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
        metrics = [dsc, 'Recall', 'Precision']  # [dsc]

    # callbacks
    callbacks = []
    if 'callbacks' in config['training']:
        if config['training']['callbacks'].get('ReduceLROnPlateau', False):
            # reduces learning rate if no improvement are seen
            learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                        patience=config['training']['callbacks']['patience'],
                                                        verbose=1,
                                                        factor=0.5,
                                                        min_lr=0.0000001)
            callbacks.append(learning_rate_reduction)

        if config['training']['callbacks'].get('EarlyStopping', False):
            # stop training if no improvements are seen
            early_stop = EarlyStopping(monitor="val_loss",
                                       mode="min",
                                       patience=int(config['training']['callbacks']['patience'] // 2),
                                       restore_best_weights=True)
            callbacks.append(early_stop)

        if config['training']['callbacks'].get('ModelCheckpoint', False):
            # saves model weights to file
            # 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
            checkpoint = ModelCheckpoint(os.path.join(training_model_folder, 'model_weights.h5'),
                                         monitor='val_loss',  # metric_dice
                                         verbose=1,
                                         save_best_only=True,
                                         mode='min',  # max
                                         save_weights_only=False)
            callbacks.append(checkpoint)

        if config['training']['callbacks'].get('TensorBoard', False):
            tensorboard_callback = TensorBoard(log_dir=logdir,
                                               histogram_freq=0,
                                               batch_size=batch_size,
                                               write_graph=True,
                                               write_grads=True,
                                               write_images=False)
            callbacks.append(tensorboard_callback)

    # callbacks = [tensorboard_callback, learning_rate_reduction, early_stop, checkpoint]

    dataset, train_transforms, val_transforms = get_data(config)
    train_images_paths, val_images_paths = dataset['train'], dataset['val']

    train_generator = DataGeneratorFromDict(train_images_paths,
                                            train_transforms,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            x_key='input', y_key='mask_img')

    val_generator = DataGeneratorFromDict(val_images_paths,
                                          val_transforms,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          x_key='input', y_key='mask_img')

    # Define model
    if architecture == 'unet':
        with strategy.scope():
            model = CustomUNet3D(tuple(list(image_shape) + [number_channels]), number_class,
                                 **cnn_params).create_model()

    elif architecture == 'vnet':
        with strategy.scope():
            model = VNet(tuple(list(image_shape) + [number_channels]), number_class,
                         **cnn_params).create_model(returns_logits=False)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/default_config_v2.json', type=str,
                        help="path/to/config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    main(config)
