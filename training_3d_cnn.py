import sys
import argparse
import json

from lib.datasets import DataManager
from lib.modality_PETCT import DataGenerator
from lib.data_loader import DataGenerator_3D_from_numpy

import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from networks.Unet import CustomUNet3D
from networks.Vnet import VNet
from networks.Layers import prelu

from losses.Loss_3d import vnet_dice_loss
from losses.Loss_3d import metric_dice as dsc

import os
from datetime import datetime


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
    resize = config['preprocessing']['resize']  # True  # not use yet
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
    opt_params = config['training']["optimizer"]["opt_params"]

    # multi gpu training strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # definition of loss, optimizer and metrics
        loss_object = vnet_dice_loss

        # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.90)
        # optimizer = tf.keras.optimizers.SGD(**opt_params)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)

        metrics = [dsc, 'BinaryCrossentropy']

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

    # Get Data
    if pp_dir is None:
        DM = DataManager(csv_path=csv_path)
        x_train, x_val, x_test, y_train, y_val, y_test = DM.get_train_val_test()

        # Define generator
        train_generator = DataGenerator(x_train, y_train,
                                        batch_size=batch_size, shuffle=shuffle, augmentation=data_augment,
                                        target_shape=image_shape, target_voxel_spacing=voxel_spacing,
                                        resize=resize, normalize=normalize, origin=origin, threshold=threshold)

        val_generator = DataGenerator(x_val, y_val,
                                      batch_size=batch_size, shuffle=False, augmentation=False,
                                      target_shape=image_shape, target_voxel_spacing=voxel_spacing,
                                      resize=resize, normalize=normalize, origin=origin, threshold=threshold)
    else:
        # mask_keys = ['mask_img_absolute', 'mask_img_relative', 'mask_img_otsu']
        mask_keys = 'mask_img_' + threshold if isinstance(threshold, str) else ['mask_img_' + el for el in threshold]
        train_generator = DataGenerator_3D_from_numpy(pp_dir, 'train',
                                                      mask_keys,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle)
        val_generator = DataGenerator_3D_from_numpy(pp_dir, 'val',
                                                    mask_keys,
                                                    batch_size=batch_size,
                                                    shuffle=False)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/default_config.json', type=str,
                        help="path/to/config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    main(config)
