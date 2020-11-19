

############
### PATH ###
############

# path/to/csv/file.csv
csv_path = "/home/oncopole/Documents/Rudy/DeepOncopole/data/DB_PATH_NIFTI.csv"

# where to save model and TensorBoard
training_model_folder = "/media/oncopole/DD 2To/RUDY_WEIGTH/training"

# path to processed data.
# If set to None, the data will be processed "on the fly"
# Else, the data will be processed and saved on this directory before training.
pp_dir = None

# path/to/pretrained/model.h5
# Set to None to train from scratch
trained_model_path = "/media/oncopole/DD 2To/RUDY_WEIGTH/training/20200911-14:27:38/trained_model_20200911-14:27:38.h5"

#####################
### PREPROCESSING ###
#####################

dim = 3
image_shape = (256, 128, 128)  # (z, y, x)
voxel_spacing = (4.0, 4.0, 4.0)  # (z, y, x), in mm

origin = 'head'  # how to set the new origin

modalities = ('pet_img', 'ct_img')
in_channels = len(modalities)
out_channels = 1  # i.e number of class. Set to 1 for sigmoid activation


# pet_pp = {keys="pet_img", a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True}
# ct_pp = {a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True}
# default_value = {'pet_img': 0.0, 'ct_img': -1000.0}


####################
### ground_truth ###
####################


#############
### MODEL ###
#############
architecture = "vnet"
cnn_params = {
    "keep_prob": 1.0,
    "keep_prob_last_layer": 0.8,
    "kernel_size": (5, 5, 5),
        "num_channels": 8,
        "num_levels": 4,
        "num_convolutions": (1, 2, 3, 3),
        "bottom_convolutions": 3,
        "activation": "relu"
    }

################
### TRAINING ###
################

epochs = 20
batch_size = 2
shuffle = True

# callback
patience = 10
ReduceLROnPlateau = False
EarlyStopping = False
ModelCheckpoint = True
TensorBoard = True
