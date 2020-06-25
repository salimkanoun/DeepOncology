#!/usr/bin/env python
# coding: utf-8

##############################################
#####               IMPORTS               ####
##############################################

import tensorflow as tf

print("Tensorflow version : %s" % tf.__version__)

import csv
import time
import random
import glob
import os
import numpy as np
from math import ceil

from class_modalities.modality_PETCT import Modality_TRAINING_PET_CT
from class_modalities.datasets import DataManager

from deeplearning_models.Unet import CustomUNet3D as CNN

from deeplearning_tools.loss_functions import Multiclass_DSC_Loss
from deeplearning_tools.loss_functions import Tumoral_DSC
from global_tools.tools import display_learning_curves

############################################
####               RULES               #####
############################################

# REQUIRED : .csv file containing all images filenames per patient
#           example : patient n°1 | PET | CT | MASK |
#                     patient n°2 | PET | CT | MASK |
#                     etc

csv_filenames = "/media/storage/projet_LYSA_TEP_3.5/TEP_CT_training_filenames.csv"

# training parameters
trained_model_path = '/media/storage/projet_LYSA_TEP_3.5/RESULTS_PETCT_4/model_09241142.h5'  # or None

# definition of modality
MODALITY = Modality_TRAINING_PET_CT()

# path folders
path_preprocessed = '/media/storage/projet_LYSA_TEP_3.5/PREPROCESS_PETCT_4'
path_results = '/media/storage/projet_LYSA_TEP_3.5/RESULTS_PETCT_4'

# generates folders
if not os.path.exists(path_preprocessed):
    os.makedirs(path_preprocessed)
if not os.path.exists(path_results):
    os.makedirs(path_results)

#############################################
####               PARAMS               #####
#############################################

# preprocess parameters
PREPROCESS_DATA = True
visualisation_preprocessed_files = True
IMAGE_SHAPE = [368, 128, 128]
PIXEL_SIZE = [4.8, 4.8, 4.8]
DATA_AUGMENT = True
RESIZE = True
NORMALIZE = True

SHUFFLE = True
labels_names = MODALITY.labels_names  # example for TEP :['Background','Lymphoma',]
labels_numbers = MODALITY.labels_numbers  # :[0,1]
ITERATIONS = 50000
BATCH_SIZE = 1

# visualisation parameters
PREDICTION_TRAINING_SET = False  # (for development or verification purpose)
PREDICTION_VALIDATION_SET = True
PREDICTION_TEST_SET = False  # (final trained model only)
saving_directives = {
    'Save history': True,
    'Save model': True
}



###########################  TRAINING   #############################################

# shuffle training data
if SHUFFLE:
    random.shuffle(preprocessed_sets['TRAIN_SET'])

# preparation of tensorflow DATASETS
train_generator = MODALITY.get_GENERATOR(preprocessed_sets['TRAIN_SET'])
train_dataset = tf.data.Dataset.from_generator(train_generator.call, train_generator.types,
                                               train_generator.shapes).batch(BATCH_SIZE)

valid_generator = MODALITY.get_GENERATOR(preprocessed_sets['VALID_SET'])
valid_dataset = tf.data.Dataset.from_generator(valid_generator.call, valid_generator.types,
                                               valid_generator.shapes).batch(BATCH_SIZE)

# MODEL PREPARATION
epochs = int(ITERATIONS / len(preprocessed_sets['TRAIN_SET']))

if trained_model_path is None:
    # GENERATE NEW MODEL
    number_channels = MODALITY.number_channels
    cnn_img_shape = tuple(IMAGE_SHAPE.copy() + [number_channels])
    model = CNN(cnn_img_shape, len(labels_names)).get_model()
else:
    # LOAD MODEL FROM .h5 FILE
    model = tf.keras.models.load_model(trained_model_path, compile=False)

# definition of loss, optimizer and metrics
loss_object = Multiclass_DSC_Loss()
metrics = [Tumoral_DSC(), tf.keras.metrics.SparseCategoricalCrossentropy(name='SCCE')]
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.1)

# TODO : generate a learning rate scheduler

model.compile(loss=loss_object, optimizer=optimizer, metrics=metrics)

# LEARNING PROCEDURE

start_tt = time.time()

history = model.fit(
    x=train_dataset,
    validation_data=valid_dataset,
    validation_steps=len(preprocessed_sets['VALID_SET']),
    epochs=epochs)

# TIMER
total_tt = time.time() - start_tt
hours = int(total_tt // 3600)
mins = int((total_tt - hours * 3600) // 60)
sec = int((total_tt - hours * 3600 - mins * 60))
print("\n\nRun time = " + str(hours) + ':' + str(mins) + ':' + str(sec) + ' (H:M:S)')

###########################  VISUALISATION   #############################################

# plot learning curves and save history
if history:

    print("Learning curves :")
    display_learning_curves(history)

    if saving_directives['Save history']:
        filename = path_results + "/history_" + time.strftime("%m%d%H%M") + ".dat"
        print("Saving history: %s" % filename)
        with open(filename, 'w') as file:
            file.write(str(history.history))

# save whole model as .h5 file
if saving_directives['Save model']:
    filename = path_results + "/model_" + time.strftime("%m%d%H%M") + ".h5"
    print("Saving model : %s" % filename)
    model.save(filename)

if PREDICTION_TRAINING_SET:
    print("Prediction on training set : /!\ use only for development or verification purpose")

    n_sample = min(20, len(preprocessed_sets['TRAIN_SET']))  # number of training imgs visualised
    random.shuffle(preprocessed_sets['TRAIN_SET'])

    filename = "/RESULTS_train_set_" + time.strftime("%m%d%H%M%S") + ".pdf"

    print("Generating predictions :")
    train_prediction_ids = MODALITY.PREDICT_MASK(data_set_ids=preprocessed_sets['TRAIN_SET'][:n_sample],
                                                 path_predictions=path_results + '/train_predictions',
                                                 model=model)

    print("\nDisplaying stats and MIP : %s" % filename)
    MODALITY.VISUALISATION_MIP_PREDICTION(path_results,
                                          data_set_ids=preprocessed_sets['TRAIN_SET'][:n_sample],
                                          pred_ids=train_prediction_ids,
                                          filename=filename)

if PREDICTION_VALIDATION_SET:
    print("Prediction on validation set :")
    # use to fine tune and evaluate model performances

    filename = "/RESULTS_valid_set_" + time.strftime("%m%d%H%M%S") + ".pdf"

    print("Generating predictions :")
    valid_prediction_ids = MODALITY.PREDICT_MASK(data_set_ids=preprocessed_sets['VALID_SET'],
                                                 path_predictions=path_results + '/valid_predictions',
                                                 model=model)

    print("\nDisplaying stats and MIP : %s" % filename)
    MODALITY.VISUALISATION_MIP_PREDICTION(path_results,
                                          data_set_ids=preprocessed_sets['VALID_SET'],
                                          pred_ids=valid_prediction_ids,
                                          filename=filename)

if PREDICTION_TEST_SET:
    print("Prediction on test set : /!\ use only on fully trained model")

    filename = "/RESULTS_test_set_" + time.strftime("%m%d%H%M%S") + ".pdf"

    print("Generating predictions :")
    test_prediction_ids = MODALITY.PREDICT_MASK(data_set_ids=preprocessed_sets['TEST_SET'],
                                                path_predictions=path_results + '/test_predictions',
                                                model=model)

    print("\nDisplaying stats and MIP : %s" % filename)
    MODALITY.VISUALISATION_MIP_PREDICTION(path_results,
                                          data_set_ids=preprocessed_sets['TEST_SET'],
                                          pred_ids=test_prediction_ids,
                                          filename=filename)
