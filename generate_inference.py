import numpy as np 
import pandas as pd 
import csv 
import SimpleITK as sitk


from experiments.exp_3d.preprocessing import *
from losses.Metrics import metric_dice

from lib.visualize import display_instance
from experiments.exp_3d.inference import Pipeline

import matplotlib.pyplot as plt

target_size = (128, 128, 256)
target_spacing = (4.0, 4.0, 4.0)
target_direction = (1,0,0,0,1,0,0,0,1)
model_path = None
target_origin = None
from_pp = False

csv_path = '/media/deeplearning/LACIE SHARE/Olivier_Morel_sarcome/inference.csv'
model = ''


df = pd.read_csv(csv_path)
dataset = df[['STUDY_UID', 'NIFTI_CT', 'NIFTI_PET']].to_dict('records')
# [{study uid : ... ,
#   nifti_ct : ... , 
#   nifti_pet : ... ,}]
print(dataset)
print("")
pipeline = Pipeline(target_size, target_spacing, target_direction, model_path=model_path, target_origin=None, from_pp=False)