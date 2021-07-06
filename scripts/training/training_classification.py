import os 
import csv 
import json 
from datetime import datetime
from classification.dataset import DataManager
from classification.preprocessing import * 
from dicom_to_cnn.tools.cleaning_dicom.folders import *
from networks.classification import classification
import tensorflow as tf
import numpy as np
#training paramaters
epochs = 12
batch_size = 256
shuffle = True 


csv_path = '/media/oncopole/d508267f-cc7d-45e2-ae24-9456e005a01f/CLASSIFICATION/classification_dataset_NIFTI_V3.csv'
training_model_folder_name = '/media/oncopole/d508267f-cc7d-45e2-ae24-9456e005a01f/CLASSIFICATION/training/train_2'
def main() : 
    
    now = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    training_model_folder = os.path.join(training_model_folder_name, now)  # '/path/to/folder'
    if not os.path.exists(training_model_folder):
        os.makedirs(training_model_folder)
            
    logdir = os.path.join(training_model_folder, 'logs')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

       # multi gpu training strategy
    strategy = tf.distribute.MirroredStrategy()

    dataset = get_data(csv_path)
    train_idx, val_idx, test_idx = dataset['train'], dataset['val'], dataset['test']
    print("TRAIN :", len(train_idx))
    print('VAL :', len(val_idx))
    print('TEST :', len(test_idx))
   #write_json_file(training_model_folder, 'test_dataset', test_idx)

    #TRAIN DATASET 
    train_path = '/media/oncopole/d508267f-cc7d-45e2-ae24-9456e005a01f/CLASSIFICATION/training/train_2/train'
    liste = os.listdir(train_path) 
    all_liste = []
    for x in liste : 
        all_liste.append(os.path.join(train_path, x))
    sorted_train = sorted(all_liste)
    print(len(sorted_train)//2)

    #VAL DATASET
    val_path = '/media/oncopole/d508267f-cc7d-45e2-ae24-9456e005a01f/CLASSIFICATION/training/train_2/val'
    liste = os.listdir(val_path) 
    all_liste = []
    for x in liste : 
        all_liste.append(os.path.join(val_path, x))
    sorted_val = sorted(all_liste)
    print(len(sorted_val)//2)

    X_train = []
    y_train = []
    X_val = []
    y_val = []

    for i in range(0, len(sorted_train),2) : 
        X_train.append(sitk.GetArrayFromImage(sitk.ReadImage(sorted_train[i+1])))
        y_train.append(get_label_from_json(sorted_train[i]))

    for i in range(0, len(sorted_val),2) : 
        X_val.append(sitk.GetArrayFromImage(sitk.ReadImage(sorted_val[i+1])))
        y_val.append(get_label_from_json(sorted_val[i]))

    X_train, y_train, X_val, y_val = np.asarray(X_train), np.asarray(y_train), np.asarray(X_val), np.asarray(y_val)
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    
    callbacks = []
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq='epoch', write_graph=True, write_images=True)
    callbacks.append(tensorboard_callback)
    with strategy.scope(): 
        model = classification(input_shape=(1024, 256, 1))
    model.summary()
    
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) #param
        model.compile(optimizer = optimizer, 
                loss={'left_arm' : 'sparse_categorical_crossentropy', 
                    'right_arm' : 'sparse_categorical_crossentropy', 
                    'head' : 'sparse_categorical_crossentropy', 
                    'legs' : 'sparse_categorical_crossentropy'}, 
                loss_weights ={'left_arm': 0.25, 'right_arm' : 0.25, 
                                'head' : 0.25, 
                                'legs': 0.25}, 
                metrics = {'left_arm': ['accuracy'], #'SparseCategoricalCrossentropy'
                            'right_arm' : ['accuracy'], 
                            'head' : ['accuracy'], 
                            'legs':['accuracy']}) #a voir pour loss

    print(model.summary())


    history = model.fit(X_train, {'head': y_train[:,0], 
                                    'legs': y_train[:,1],
                                    'right_arm' : y_train[:,2],
                                    'left_arm' : y_train[:,3], }, 
                            validation_data = (X_val, {'head': y_val[:,0], 
                                    'legs': y_val[:,1],
                                    'right_arm' : y_val[:,2],
                                    'left_arm' : y_val[:,3] ,
                                    }),
                            epochs=epochs,
                            callbacks=callbacks,  # initial_epoch=0,
                            verbose=1
                            )
    model.save(os.path.join(training_model_folder, 'trained_model_{}.h5'.format(now)))
    model.save(os.path.join(training_model_folder, 'trained_model_{}'.format(now)))
    
    
if __name__ == "__main__":
    main()