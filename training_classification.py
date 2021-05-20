import SimpleITK as sitk 
import numpy as np 
import matplotlib.pyplot as plt 
import json 
import csv 
import os 
import pandas as pd 
import tensorflow as tf 
from sklearn.model_selection import train_test_split 
from classification.tools.prepare_csv_from_json import *
from classification.pre_process.Preprocessing import Preprocessing 
from networks.classification import *
from classification.tools.generate_2d_image import *


json_path = '/media/deeplearning/78ca2911-9e9f-4f78-b80a-848024b95f92/result.json'
image_directory = 'path/to/directory/image_png'
model_directory = 'path/save/model'

objet = JSON_TO_CSV(json_path)
csv_path = objet.result_csv(image_directory, model_directory)
print(objet.csv_path)

preprocessing_objet = Preprocessing(objet.csv_path)
X, y = preprocessing_objet.normalize_encoding_dataset()

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42, test_size = 0.15) #random state 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = 42, test_size = 0.15)
print("size of X_train : ", X_train.shape)
print("size of y_train : ",y_train.shape)
print("")
print("size of X_test : ", X_test.shape)
print("size of y_test : ",y_test.shape)
print("")
print("size of X_val : ", X_val.shape)
print("size of y_val : ",y_val.shape)


model = classification(input_shape=(503, 136, 1))
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


now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
training_model_folder = os.path.join(model_directory, now)  # '/path/to/folder'
    if not os.path.exists(training_model_folder):
        os.makedirs(training_model_folder)
            
log_dir = os.path.join(training_model_folder, 'logs')
if not os.path.exists(logdir):
    os.makedirs(logdir)


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch', write_graph=True, write_images=True)

#fit 

history = model.fit(X_train, {'head': y_train[:,0], 
                                    'leg': y_train[:,1],
                                    'right_arm' : y_train[:,2],
                                    'left_arm' : y_train[:,3] ,
                                    }, 
                                    
                        epochs = 20, 
                        batch_size = 256, 
                        verbose = 1, 
                        validation_data = (X_val, {'head': y_val[:,0], 
                                    'leg': y_val[:,1],
                                    'right_arm' : y_val[:,2],
                                    'left_arm' : y_val[:,3] ,
                                    }),
                        callbacks=[tensorboard_callback])



hist_df = pd.DataFrame(history.history)
hist_json_file = 'history.json'
with open(model_directory+'/'+hist_json_file, mode = 'w') as f : 
    hist_df.to_json(f)
print("history saved")

#save 
model.save(model_directory + '/' + 'classification_model', save_format='h5')
print('model saved as .h5')
model.save(model_directory+'/'+ 'classification_model_fold')
print('model saved')
