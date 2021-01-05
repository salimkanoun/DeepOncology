import argparse
import json

import os
from datetime import datetime

from lib.data_loader import DataGeneratorFromDict
# from lib.transforms import *
from experiments.exp_3d.preprocessing import *

import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from networks.Vnet import VNet

from losses.Loss import custom_robust_loss
from losses.Loss import metric_dice 

#from lib.utils import read_cfg



csv_path = ''
pp_dir = ''

#### PRE PROCESSING #####

modalities = ('pet_img', 'ct_img')
mode = ['binary', 'probs', 'mean_probs'][0]
method = ['otsu', 'absolute', 'relative', 'otsu_abs'][0]

#[if mode = binary & method = relative : t_val = 0.42
#if mode = binary & method = absolute : t_val = 2.5, 
#else : don't need tval]
tval = ''
target_size = (128, 128, 256)
target_spacing = (4.0, 4.0, 4.0)
target_direction = (1,0,0,0,1,0,0,0,1)

from_pp=False
cache_pp=False
pp_flag=''

##### TENSORFLOW #####


trained_model_path = '' #If None, train from scratch 
training_model_folder = ''

#training paramaters
epochs = 100
batch_size = 2
shuffle = True 

#callbacks
patience = 10
ReduceLROnPlateau = False
EarlyStopping = False
ModelCheckpoint = True
TensorBoard = True

##### ARCHITECTURE #####
#model
architecture = 'vnet'

#parameters
image_shape= (256, 128, 128)
in_channels= len(modalities)
out_channels= 1
channels_last=True
keep_prob= 1.0
keep_prob_last_layer= 0.8
kernel_size= (5, 5, 5)
num_channels= 8
num_levels= 4
num_convolutions= (1, 2, 3, 3)
bottom_convolutions= 3
activation= "relu"
activation_last_layer= 'sigmoid'





def main():
    # path
    now = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    
    training_model_folder = os.path.join(training_model_folder, now)  # '/path/to/folder'
    if not os.path.exists(training_model_folder):
        os.makedirs(training_model_folder)
    logdir = os.path.join(training_model_folder, 'logs')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # saving the config in the result folder
    #copyfile(cfg['cfg_path'], os.path.join(training_model_folder, 'config.py'))


    # multi gpu training strategy
    strategy = tf.distribute.MirroredStrategy()

    # Get Data path and transforms
    #get_data function from exp_3D/preprocessing 
    dataset, train_transforms, val_transforms = get_data(pp_dir, csv_path, modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin=None , data_augmentation=True, from_pp=from_pp, cache_pp=cache_pp, pp_flag=pp_flag)
    #dataset = dict('train' : [{ct pet mask}, {},] 
    #                'test' : [{ct pet mask}, {},] 
    #                 'val' : [{ct pet mask}, {}, ])
    #train, val_transforms = list of transformers to applied (preprocessing)


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

    with strategy.scope():
        # definition of loss, optimizer and metrics
        loss_object = custom_robust_loss(dim=3)
        optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
        dsc = metric_dice(dim=3, vnet=True)
        metrics = [dsc, 'Recall', 'Precision']  # [dsc]

    # callbacks
    callbacks = []
    if ReduceLROnPlateau == True :
        # reduces learning rate if no improvement are seen
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                    patience=patience ,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.0000001)
        callbacks.append(learning_rate_reduction)

    if EarlyStopping == True :
        # stop training if no improvements are seen
        early_stop = EarlyStopping(monitor="val_loss",
                                   mode="min",
                                   patience=int(patience // 2),
                                   restore_best_weights=True)
        callbacks.append(early_stop)

    if ModelCheckpoint == True :
        # saves model weights to file
        # 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpoint = ModelCheckpoint(os.path.join(training_model_folder, 'model_weights.h5'),
                                     monitor='val_loss',  # metric_dice
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',  # max
                                     save_weights_only=False)
        callbacks.append(checkpoint)

    if TensorBoard == True :
        tensorboard_callback = TensorBoard(log_dir=logdir,
                                           histogram_freq=0,
                                           batch_size=batch_size,
                                           write_graph=True,
                                           write_grads=True,
                                           write_images=False)
        callbacks.append(tensorboard_callback)

    # Define model
    if architecture.lower() == 'vnet':
        with strategy.scope():
            model = VNet(image_shape,
                 in_channels,
                 out_channels,
                 channels_last,
                 keep_prob,
                 keep_prob_last_layer,
                 kernel_size,
                 num_channels,
                 num_levels,
                 num_convolutions,
                 bottom_convolutions,
                 activation,
                 activation_last_layer).create_model()
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


    #SAVE HISTORY ? 

    
    # whole model saving
    model.save(os.path.join(training_model_folder, 'trained_model_{}.h5'.format(now)))


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-c", "--config", default='config/config_3d.py', type=str,
    #                    help="path/to/config.py")
    #args = parser.parse_args()

    #config = read_cfg(args.config)
    #config['cfg_path'] = args.config

    main()
