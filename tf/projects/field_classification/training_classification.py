import os 
from datetime import datetime
import tensorflow as tf
from tf.project.field_classification.lib.datasets import * 
from dicom_to_cnn.tools.cleaning_dicom.folders import *
from tf.networks.classification import classification
from tf.project.field_classification.data_loader import DataGeneratorFromDict
import tensorflow as tfw


#training paramaters
epochs = 12
batch_size = 256
shuffle = True 
x_keys = 'ct_img'
y_keys = ['upper_limit', 'lower_limit', 'right_arm', 'left_arm']


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
    strategy = tfw.distribute.MirroredStrategy()

    dataset = get_data(csv_path)
    train_idx, val_idx, test_idx = dataset['train'], dataset['val'], dataset['test']
    print("TRAIN :", len(train_idx))
    print('VAL :', len(val_idx))
    print('TEST :', len(test_idx))
    write_json_file(training_model_folder, 'test_dataset', test_idx)

    train_generator = DataGeneratorFromDict(train_idx, batch_size, shuffle, x_keys, y_keys)
    val_generator = DataGeneratorFromDict(val_idx, batch_size, shuffle, x_keys, y_keys)

   
    callbacks = []
    tensorboard_callback = tfw.keras.callbacks.TensorBoard(log_dir=logdir, update_freq='epoch', write_graph=True, write_images=True)
    callbacks.append(tensorboard_callback)
    with strategy.scope(): 
        model = classification(input_shape=(1024, 256, 1))
    model.summary()
    
    with strategy.scope():
        optimizer = tfw.keras.optimizers.Adam(learning_rate=1e-3) #param
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


    history = model.fit(train_generator, 
                            steps_per_epoch=len(train_generator),
                            validation_data = val_generator,
                            validation_steps=len(val_generator),
                            epochs=epochs,
                            callbacks=callbacks,  # initial_epoch=0,
                            verbose=1
                            )

    model.save(os.path.join(training_model_folder, 'trained_model_{}.h5'.format(now)))
    model.save(os.path.join(training_model_folder, 'trained_model_{}'.format(now)))
    
    
if __name__ == "__main__":
    main()