from class_modalities.datasets import DataManager
from class_modalities.modality_PETCT import DataGenerator

import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from deeplearning_models.Unet import CustomUNet3D
from deeplearning_models.Vnet import VNet

from deeplearning_tools.loss_functions import Tumoral_DSC, Multiclass_DSC_Loss

import os
from datetime import datetime

# path
now = datetime.now().strftime("%Y%m%d-%H%M%S")

data_path = r'C:\\Users\\Rudy\\Documents\\Thales_stage\\data\\nifti_scan'  # '/path/to/data'

trained_model_path = None  # if None, trained from scratch
training_model_folder = os.path.join(r'C:\\Users\\Rudy\\Documents\\Thales_stage\\training', now)  # '/path/to/folder'
if not os.path.exists(training_model_folder):
    os.makedirs(training_model_folder)
logdir = os.path.join(training_model_folder, 'logs')
if not os.path.exists(logdir):
    os.makedirs(logdir)

# generate_MIP_prediction = True
# result_path = r'C:\\Users\\Rudy\\Documents\\Thales_stage\\training'

# PET CT scan params
image_shape = (128, 64, 64)  # (368, 128, 128)  # (z, y, x)
number_channels = 2
voxel_spacing = (4.8, 4.8, 4.8)  # in millimeter, (z, y, x)
data_augment = True  # for training dataset only
resize = True  # not use yet
normalize = True  # whether or not to normalize the inputs
number_class = 2

# CNN params
architecture = 'vnet'  # 'unet' or 'vnet'

if architecture == 'unet':
    cnn_params = {'filters': (8, 16, 32),  # (8, 16, 32, 64, 128)
                  'kernel': (3, 3, 3),
                  'activation': tf.keras.layers.LeakyReLU(),
                  'padding': 'same',
                  'pooling': (2, 2, 2)}

elif architecture == 'vnet':
    cnn_params = {'keep_prob': 1.0,
                  'kernel_size': (5, 5, 5),
                  'num_channels': 16,
                  'num_levels': 4,
                  'num_convolutions': (1, 2, 3, 3),
                  'bottom_convolutions': 3,
                  'activation_fn': tf.keras.layers.ReLU()}
else:
    raise ValueError('Architecture ' + architecture + ' not supported. Please ' +
                     'choose one of unet|vnet.')

# Training params
epochs = 3
batch_size = 1
shuffle = True

# definition of loss, optimizer and metrics
loss_object = Multiclass_DSC_Loss()
metrics = [Tumoral_DSC(), tf.keras.metrics.SparseCategoricalCrossentropy(name='SCCE')]
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)

# callback
patience = 50

# reduces learning rate if no improvement are seen
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=patience,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0000001)

# stop training if no improvements are seen
early_stop = EarlyStopping(monitor="val_loss",
                           mode="min",
                           patience=int(patience // 2),
                           restore_best_weights=True)

# saves model weights to file
checkpoint = ModelCheckpoint(os.path.join(training_model_folder, 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             save_weights_only=True)

tensorboard_callback = TensorBoard(log_dir=logdir,
                                   histogram_freq=0,
                                   batch_size=batch_size,
                                   write_graph=True,
                                   write_grads=True,
                                   write_images=False)

callbacks = [tensorboard_callback, learning_rate_reduction, early_stop, checkpoint]

# Get Data
DM = DataManager(base_path=data_path)
x_train, x_val, x_test, y_train, y_val, y_test = DM.get_train_val_test()

# Define generator
train_generator = DataGenerator(x_train, y_train,
                                batch_size=batch_size, shuffle=shuffle, augmentation=data_augment,
                                target_shape=image_shape, target_voxel_spacing=voxel_spacing,
                                resize=resize, normalize=normalize)

val_generator = DataGenerator(x_val, y_val,
                              batch_size=batch_size, shuffle=False, augmentation=False,
                              target_shape=image_shape, target_voxel_spacing=voxel_spacing,
                              resize=resize, normalize=normalize)

# Define model
if architecture == 'unet':
    model = CustomUNet3D(tuple(list(image_shape) + [number_channels]), number_class,  **cnn_params).create_model()

elif architecture == 'vnet':
    model = VNet(tuple(list(image_shape) + [number_channels]), number_class, **cnn_params).create_model()
else:
    raise ValueError('Architecture ' + architecture + ' not supported. Please ' +
                     'choose one of unet|vnet.')

if trained_model_path is not None:
    model.load_weights(trained_model_path)


model.compile(loss=loss_object, optimizer=optimizer, metrics=metrics)

# training model
history = model.fit_generator(generator=train_generator,
                              validation_data=val_generator,
                              epochs=epochs,
                              steps_per_epoch=len(train_generator),
                              validation_steps=len(val_generator),
                              callbacks=callbacks,
                              verbose=1
                              )

# save model/history/performance
model.save(os.path.join(training_model_folder, 'trained_model_{}.h5').format(now))

# # save hyper parameter and train, val, test performance
# header = ['date', 'architecture', 'filters', 'kernel', 'activation', 'padding', 'pooling']
# row = [now, architecture, filters, kernel, activation, padding, pooling]
