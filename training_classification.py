import SimpleITK as sitk 
import numpy as np 
import matplotlib.pyplot as plt 
import json 
import csv 
import os 
import pandas as pd 
import tensorflow as tf 
from sklearn.model_selection import train_test_split 

from classification.pre_process.Prep_CSV import Prep_CSV
from classification.pre_process.Preprocessing import Preprocessing 
from classification.model.classification import *
from utils.modality_CT import *


json_path = '/media/deeplearning/78ca2911-9e9f-4f78-b80a-848024b95f92/result.json'
nifti_directory = '/media/deeplearning/78ca2911-9e9f-4f78-b80a-848024b95f92'
objet = Prep_CSV(json_path)
objet.result_csv(nifti_directory)
print(objet.csv_result_path)

prep_objet = Preprocessing(objet.csv_result_path)
X, y = prep_objet.normalize_encoding_dataset()

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


log_dir = '/home/deeplearning/Deep_Learning_result/classic_model_test/logs_21_10_2020'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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
                        #validation_split= 0.20,
                        validation_data = (X_val, {'head': y_val[:,0], 
                                    'leg': y_val[:,1],
                                    'right_arm' : y_val[:,2],
                                    'left_arm' : y_val[:,3] ,
                                    }),
                        callbacks=[tensorboard_callback])



hist_df = pd.DataFrame(history.history)
hist_json_file = 'history.json'
with open('/home/deeplearning/Deep_Learning_result/classic_model_test'+'/'+hist_json_file, mode = 'w') as f : 
    hist_df.to_json(f)
print("history saved")

#save 
folder = '/home/deeplearning/Deep_Learning_result/classic_model_test'
if not os.path.exists(folder) : 
    os.makedirs(folder)
model.save(folder + '/' + 'classic_model', save_format='h5')
