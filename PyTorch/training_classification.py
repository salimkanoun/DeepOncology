import os
from datetime import datetime
from Projects.Classification.train_classification import train_classification

json_path = '../../torch_files/result.json' #path from PyTorch
nifti_directory = '/media/m-056285chu-toulousefr/c6a4c10e-9316-4b95-8c6e-1d87e1dce435/lysarc/NIFTI/'
csv_directory = '/media/m-056285chu-toulousefr/c6a4c10e-9316-4b95-8c6e-1d87e1dce435/lysarc/'

training_model_folder = '../../pytorch_models' #base path to save info network, trained mode (.h5), ...
now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
training_model_folder = os.path.join(training_model_folder, now)  # '/path/to/folder'
if not os.path.exists(training_model_folder):
    os.makedirs(training_model_folder)
    #if the directory doesn't exist creates it

train_classification(json_path, nifti_directory, csv_directory, training_model_folder)