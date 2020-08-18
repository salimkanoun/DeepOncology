import sys
import json

from class_modalities.datasets import DataManager
from class_modalities.modality_PETCT_2D import DataGenerator

import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from deeplearning_models.DenseXnet import DenseXNet as cnn
from deeplearning_models.Layers import prelu

from deeplearning_tools.Loss_2d import densexnet_loss, metric_dice

import os
from datetime import datetime

# path
now = datetime.now().strftime("%Y%m%d-%H:%M:%S")

# import config file
if len(sys.argv) == 2:
    config_name = sys.argv[1]
else:
    config_name = 'config/default_config_2d.json'

with open(config_name) as f:
    config = json.load(f)

# path
csv_path = config['path']['csv_path']

trained_model_path = config['path']['trained_model_path']  # if None, trained from scratch
training_model_folder = os.path.join(config['path']['training_model_folder'], now)  # '/path/to/folder'
if not os.path.exists(training_model_folder):
    os.makedirs(training_model_folder)
logdir = os.path.join(training_model_folder, 'logs')
if not os.path.exists(logdir):
    os.makedirs(logdir)

# PET CT scan params
image_shape = tuple(config['preprocessing']['image_shape'])  # (300, 128, 128)  # (z, y, x)
in_channels, out_channels = config['model']['in_channels'], config['model']['out_channels']  # 6, 1
voxel_spacing = tuple(config['preprocessing']['voxel_spacing'])  # (4.8, 4.8, 4.8)  # in millimeter, (z, y, x)
data_augment = config['preprocessing']['data_augment']  # True  # for training dataset only
resize = config['preprocessing']['resize']  # True  # not use yet
normalize = config['preprocessing']['normalize']  # True  # whether or not to normalize the inputs

# CNN params
architecture = config['model']['architecture']

cnn_params = config['model'][architecture]['cnn_params']
# transform list to tuple
for key, value in cnn_params.items():
    if isinstance(value, list):
        cnn_params[key] = tuple(value)

# Training params
epochs = config['training']['epochs']
batch_size = config['training']['batch_size']
shuffle = config['training']['shuffle']

# multi gpu training strategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # definition of loss, optimizer and metrics
    loss_object = densexnet_loss

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    metrics = [metric_dice, 'BinaryCrossentropy', 'Recall', 'Precision']
    # tf.keras.metrics.Recall(),

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
        checkpoint = ModelCheckpoint(os.path.join(training_model_folder, 'model_weights.hdf5'),
                                     # 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     save_weights_only=True)
        callbacks.append(checkpoint)

    if config['training']['callbacks'].get('TensorBoard', False):
        tensorboard_callback = TensorBoard(log_dir=logdir,
                                           histogram_freq=0,
                                           batch_size=batch_size,
                                           write_graph=True,
                                           write_grads=True,
                                           write_images=False)
        callbacks.append(tensorboard_callback)

# Get Data
DM = DataManager(csv_path=csv_path)
train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)

# Define generator
train_generator = DataGenerator(train_images_paths,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                augmentation=data_augment,
                                target_shape=image_shape,
                                target_voxel_spacing=voxel_spacing)

val_generator = DataGenerator(val_images_paths,
                              batch_size=batch_size,
                              shuffle=False,
                              augmentation=False,
                              target_shape=image_shape,
                              target_voxel_spacing=voxel_spacing)


# Define model
with strategy.scope():
    model = cnn(image_shape[1:], in_channels, out_channels, **cnn_params).create_model()
    model.compile(loss=loss_object, optimizer=optimizer, metrics=metrics)

    if trained_model_path is not None:
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
                    verbose=1,
                    workers=2, use_multiprocessing=True
                    )

# whole model saving
model.save(os.path.join(training_model_folder, 'trained_model_{}.h5'.format(now)))
