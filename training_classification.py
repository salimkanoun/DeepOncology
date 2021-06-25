import os 
import csv 
import json 
import datetime
from classification.dataset import DataManager
from classification.preprocessing import * 
from dicom_to_cnn.tools.cleaning_dicom.folders import *
from networks.classification import classification
import tensorflow as tf

#training paramaters
epochs = 20
batch_size = 256
shuffle = True 





csv_path = ''
training_model_folder_name = ''
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



    dataset = get_data(csv_path)
    train_idx, val_idx, test_idx = dataset['train'], dataset['val'], dataset['test']

    print("TRAIN :", len(train_idx))
    print('VAL :', len(val_idx))
    print('TEST :', len(test_idx))
    write_json_file(training_model_folder, 'test_dataset', test_idx)


    train_generator = DataGeneratorFromDict(train_idx,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                x_key='input', y_key='mask_img')

    val_generator = DataGeneratorFromDict(val_idx,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            x_key='input', y_key='mask_img')

    def make_train_gen():
        return train_generator 
    def make_val_gen():
        return val_generator

    training_dataset = tf.data.Dataset.from_generator(make_train_gen, (tf.float16, tf.uint8))
    training_dataset = training_dataset.cache(cachedir_train).repeat()
    val_dataset = tf.data.Dataset.from_generator(make_val_gen, (tf.float16, tf.uint8))
    val_dataset = val_dataset.cache(cachedir_val).repeat()

    callbacks = []
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq='epoch', write_graph=True, write_images=True)
    callbacks.append(tensorboard_callback)

    model = classification(input_shape=(1024, 256, 1))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) #param
    model.compile(optimizer = optimizer, 
            loss={'left_arm' : 'sparse_categorical_crossentropy', 
                'right_arm' : 'sparse_categorical_crossentropy', 
                'head' : 'sparse_categorical_crossentropy', 
                'leg' : 'sparse_categorical_crossentropy'}, 
            loss_weights ={'left_arm': 0.25, 'right_arm' : 0.25, 
                            'head' : 0.25, 
                            'leg': 0.25}, 
            metrics = {'left_arm': ['accuracy'], #'SparseCategoricalCrossentropy'
                        'right_arm' : ['accuracy'], 
                        'head' : ['accuracy'], 
                        'leg':['accuracy']}) #a voir pour loss

    print(model.summary())


    history = model.fit(training_dataset,
                            steps_per_epoch=len(train_generator)//batch_size,
                            validation_data=val_dataset,
                            validation_steps=len(val_generator)//batch_size,
                            epochs=epochs,
                            callbacks=callbacks,  # initial_epoch=0,
                            verbose=1
                            )
    model.save(os.path.join(training_model_folder, 'trained_model_{}.h5'.format(now)))
    model.save(os.path.join(training_model_folder, 'trained_model_{}'.format(now)))

    
if __name__ == "__main__":
    main()