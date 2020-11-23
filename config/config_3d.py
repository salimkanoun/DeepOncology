from math import pi
import SimpleITK as sitk

############
### PATH ###
############

# path/to/csv/file.csv
csv_path = "/home/oncopole/Documents/Rudy/DeepOncopole/data/DB_PATH_NIFTI.csv"

# where to save model and TensorBoard
training_model_folder = "/media/oncopole/DD 2To/RUDY_WEIGTH/training"

# path/to/pretrained/model.h5
# Set to None to train from scratch
trained_model_path = "/media/oncopole/DD 2To/RUDY_WEIGTH/training/20200911-14:27:38/trained_model_20200911-14:27:38.h5"

# path to processed data.
# If set to None, the data will be processed "on the fly"
# Else, the data will be processed and saved on this directory before training.
# folder architecture pp_dir/<subset>/<study_uid>/<filename>
pp_dir = None
pp_filenames_dict = {'pet_img': 'nifti_PET.nii',
                     'ct_img': 'nifti_CT.nii',
                     'mask_img': 'nifti_MASK.nii'}

#####################
### PREPROCESSING ###
#####################

dim = 3  # Must be 3
image_shape = (256, 128, 128)  # (z, y, x)
voxel_spacing = (4.0, 4.0, 4.0)  # (z, y, x), in mm

origin = 'head'  # how to set the new origin

modalities = ('pet_img', 'ct_img')
assert 'pet_img' in modalities
in_channels = len(modalities)
out_channels = 1  # i.e number of class. Except for binary classification with sigmoid activation, then set to 1.

pet_pp = dict(a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True)
ct_pp = dict(a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True)
interpolator = {'pet_img': sitk.sitkLinear,
                'ct_img': sitk.sitkLinear,
                'output': sitk.sitkLinear}
default_value = {'pet_img': 0.0, 'ct_img': -1000.0, 'mask_img': 0}
pp_kwargs = dict(pet_img=pet_pp,
                 ct_img=ct_pp,
                 interpolator=interpolator,
                 default_value=default_value)

####################
### ground_truth ###
####################
mode = ['binary', 'probs', 'mean_probs'][0]
method = ['otsu', 'absolute', 'relative', 'otsu_abs'][3]
tvals_probs = dict(absolute=dict(lower=2.0, upper=4.0, mu=2.5, std=0.5),
                   relative=dict(lower=0.33, upper=0.60, mu=0.42, std=0.06))
tvals_binary = dict(absolute=2.5,
                    relative=0.42)


#############
### MODEL ###
#############
architecture = "vnet"
cnn_kwargs = {
    "keep_prob": 1.0,
    "keep_prob_last_layer": 0.8,
    "kernel_size": (5, 5, 5),
    "num_channels": 8,
    "num_levels": 4,
    "num_convolutions": (1, 2, 3, 3),
    "bottom_convolutions": 3,
    "activation": "relu",
    "activation_last_layer": 'sigmoid'
}

################
### TRAINING ###
################

epochs = 20
batch_size = 2
shuffle = True
data_augmentation = True
da_kwargs = dict(translation=10, scaling=0.1, rotation=(pi / 60, pi / 30, pi / 60),
                 interpolator=interpolator,
                 default_value=default_value)

# callback
patience = 10
ReduceLROnPlateau = False
EarlyStopping = False
ModelCheckpoint = True
TensorBoard = True
