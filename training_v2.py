import json
import numpy as np 
import os
from datetime import datetime
import time 

from lib.data_loader import DataGeneratorFromDict
# from lib.transforms import *
from experiments.exp_3d.preprocessing import *

import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from networks.Vnet import VNet

from losses.Loss import custom_robust_loss, loss_dice
from losses.Loss import metric_dice 

from library_dicom.dicom_processor.tools.folders import *

#from lib.utils import read_cfg



csv_path = '/media/oncopole/DD 2To/SEGMENTATION/SEG_NIFTI_PATH.csv'
pp_dir = None 

#### PRE PROCESSING #####

modalities = ('pet_img', 'ct_img')
mode = ['binary', 'probs', 'mean_probs'][1]
method = ['otsu', 'absolute', 'relative', 'otsu_abs'][0:3]

# To run on a mean ground-truth of multiple run
# mode, method = 'probs', 
#method = ['otsu', 'absolute', 'relative']

#[if mode = binary & method = relative : t_val = 0.42
#if mode = binary & method = absolute : t_val = 2.5, 
#else : don't need tval]
tval_rel = 0.42
tval_abs = 2.5
target_size = (128, 128, 256)
target_spacing = (4.0, 4.0, 4.0)
target_direction = (1,0,0,0,1,0,0,0,1)

from_pp=False
cache_pp=False
pp_flag=''

##### TENSORFLOW #####


trained_model_path = '/media/oncopole/DD 2To/RUDY_WEIGTH/training/20201113-09:54:23/model_weights.h5' #If None, train from scratch 
training_model_folder_name = '/media/oncopole/DD 2To/SEGMENTATION/training'

#training paramaters
epochs = 100
batch_size = 2
shuffle = True 

#callbacks
patience = 10
ReduceLROnPlateau_val = True 
EarlyStopping_val = False
ModelCheckpoint_val = True
TensorBoard_val = True


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


def main() : 
    now = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    training_model_folder = os.path.join(training_model_folder_name, now)  # '/path/to/folder'
    if not os.path.exists(training_model_folder):
        os.makedirs(training_model_folder)
            
    logdir = os.path.join(training_model_folder, 'logs')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    #Cache with path
    cachedir_train = os.path.join(training_model_folder, 'cache_train')
    cachedir_val = os.path.join(training_model_folder, 'cache_val')
    if not os.path.exists(cachedir_train):
        os.makedirs(cachedir_train)
    if not os.path.exists(cachedir_val):
        os.makedirs(cachedir_val)

        # multi gpu training strategy
    strategy = tf.distribute.MirroredStrategy()

        # Get Data path and transforms
        #get_data function from exp_3D/preprocessing 
    dataset, train_transforms, val_transforms = get_data(pp_dir, csv_path, modalities, mode, method, tval_rel, target_size, target_spacing, target_direction, target_origin=None , data_augmentation=True, from_pp=from_pp, cache_pp=cache_pp, pp_flag=pp_flag)
        #dataset = dict('train' : [{ct pet mask}, {},] 
        #                'test' : [{ct pet mask}, {},] 
        #                 'val' : [{ct pet mask}, {}, ])
        #train, val_transforms = list of transformers to applied (preprocessing)


    train_images_paths, val_images_paths, test_images_paths = dataset['train'], dataset['val'], dataset['test']
    print("TRAIN :", len(train_images_paths))
    print('VAL :', len(val_images_paths))
    print('TEST :', len(test_images_paths))
    
    write_json_file(training_model_folder, 'test_dataset', test_images_paths)

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

    def make_train_gen():
        return train_generator 
    def make_val_gen():
        return val_generator


    training_dataset = tf.data.Dataset.from_generator(make_train_gen, (tf.float16, tf.float16))
    training_dataset = training_dataset.cache(cachedir_train).repeat()
    val_dataset = tf.data.Dataset.from_generator(make_val_gen, (tf.float16, tf.float16))
    val_dataset = val_dataset.cache(cachedir_val).repeat()

    with strategy.scope():
            # definition of loss, optimizer and metrics
        #loss_object = custom_robust_loss(dim=3)
        loss_object = loss_dice(dim=3, vnet=True)
        #loss_object = tf.keras.losses.BinaryCrossentropy()
        #optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
        optimizer = tf.optimizers.SGD(learning_rate = 0.001, momentum = 0.9)
        dsc = metric_dice(dim=3, vnet=True)
        metrics = [dsc, 'Recall', 'Precision']  # [dsc]


        
        # callbacks
    callbacks = []
    if ReduceLROnPlateau_val == True :
            # reduces learning rate if no improvement are seen
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                        patience=patience ,
                                                        verbose=1,
                                                        factor=0.5,
                                                        min_lr=0.0000001)
        callbacks.append(learning_rate_reduction)

    if EarlyStopping_val == True :
            # stop training if no improvements are seen
        early_stop = EarlyStopping(monitor="val_loss",
                                    mode="min",
                                    patience=int(patience // 2),
                                    restore_best_weights=True)
        callbacks.append(early_stop)

    if ModelCheckpoint_val == True :
            # saves model weights to file
            # 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpoint = ModelCheckpoint(os.path.join(training_model_folder, 'model_weights.h5'),
                                        monitor='val_loss',  # metric_dice
                                        verbose=1,
                                        save_best_only=True,
                                        mode='min',  # max
                                        save_weights_only=False)
        callbacks.append(checkpoint)

    if TensorBoard_val == True :
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


    history = model.fit(training_dataset,
                            steps_per_epoch=len(train_generator)//batch_size,
                            validation_data=val_dataset,
                            validation_steps=len(val_generator)//batch_size,
                            epochs=epochs,
                            callbacks=callbacks,  # initial_epoch=0,
                            verbose=1
                            )


        #SAVE HISTORY ? 

        
        # whole model saving
    model.save(os.path.join(training_model_folder, 'trained_model_{}.h5'.format(now)))
        
    
if __name__ == "__main__":
    main()