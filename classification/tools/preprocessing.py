from classification.dataset import DataManager
import numpy as np 
import SimpleITK as sitk 
from dicom_to_cnn.model.reader.Nifti import Nifti 
from dicom_to_cnn.model.post_processing.mip.MIP_Generator import MIP_Generator 
import tensorflow as tf
import json
import os 

def get_data(csv_path):

    DM = DataManager(csv_path)
    train, val, test = DM.get_train_val_test()
    dataset = dict()
    dataset['train'], dataset['val'], dataset['test'] = train, val, test 
    return dataset 

def encoding_instance(dictionnary):
    """encoding label 

    Args:
        dictionnary ([dict]): ['right arm': value, 'left_arm':value, 'head':value, 'leg':value]

    Returns:
        [list]: [return a list with encoded labels ]
    """
    label = []
    
        #upper Limit 
    if dictionnary['upper_limit'] == 'Vertex' : 
        label.append(0)
    if dictionnary['upper_limit'] == 'Eye'  or dictionnary['upper_limit'] == 'Mouth' : 
        label.append(1)

        #lower Limit
    if dictionnary['lower_limit'] == 'Hips' : 
        label.append(0)
    if dictionnary['lower_limit'] == 'Knee': 
        label.append(1)
    if dictionnary['lower_limit'] == 'Foot':
        label.append(2)

        #right Arm 
    if dictionnary['right_arm'] == 'down' : 
        label.append(0)
    if dictionnary['right_arm'] == 'up' : 
        label.append(1)

        #left Arm 
    if dictionnary['left_arm'] == 'down' : 
        label.append(0)
    if dictionnary['left_arm'] == 'up' : 
        label.append(1)

    return label


def prepare_batch(img_dict):
    resampled_array = Nifti(img_dict['ct_img']).resample(shape_matrix=(256, 256, 1024), shape_physic=(700, 700, 2000))
    resampled_array[np.where(resampled_array < 500)] = 0 #500 UH
    normalize = resampled_array[:,:,:,]/np.max(resampled_array)
    mip_generator = MIP_Generator(normalize)
    mip = mip_generator.project(angle=0)
    mip = np.expand_dims(mip, -1)
    label = encoding_instance(img_dict)

    return mip, label

def get_label_from_json(json_path):
    with open(json_path) as json_file : 
        reader = json.load(json_file)
    return reader


