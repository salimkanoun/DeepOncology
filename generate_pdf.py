import sys
import json

import os
import ntpath
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from class_modalities.datasets import DataManager
from class_modalities.modality_PETCT import DataGenerator
import collections


# read arg values
if len(sys.argv) == 2:
    config_name = 'config/default_config.json'
    MIP_folder = sys.argv[1]
elif len(sys.argv) == 3:
    config_name = sys.argv[1]
    MIP_folder = sys.argv[2]
else:
    config_name = 'config/default_config.json'
    MIP_folder = '/home/salim/Documents/DeepOncopole/MIP_dataset'

# read config file
with open(config_name) as f:
    config = json.load(f)


# path
csv_path = config['path']['csv_path']  # /media/salim/DD 2To/AHL2011_NIFTI/AHL2011_PET0_NIFTI.csv'

# PET CT scan params
image_shape = tuple(config['preprocessing']['image_shape'])  # (128, 64, 64)  # (368, 128, 128)  # (z, y, x)
number_channels = config['preprocessing']['number_channels']
voxel_spacing = tuple(config['preprocessing']['voxel_spacing'])  # (4.8, 4.8, 4.8)  # in millimeter, (z, y, x)
data_augment = False
resize = config['preprocessing']['resize']  # True  # not use yet
normalize = config['preprocessing']['normalize']  # True  # whether or not to normalize the inputs
number_class = config['preprocessing']['number_class']  # 2
threshold = config['preprocessing']['threshold']

batch_size = 1
shuffle = False

# plot params
transparency = 0.7

color_CT = plt.cm.gray
color_PET = plt.cm.plasma
color_MASK = plt.cm.Greys
color_MASK.set_bad('white', 0.0)

# Get Data
DM = DataManager(csv_path=csv_path)
dataset = collections.defaultdict(dict)
dataset['train']['x'], dataset['val']['x'], dataset['test']['x'], dataset['train']['y'], dataset['val']['y'], \
dataset['test']['y'] = DM.get_train_val_test()

# for subset_type in dataset.keys():
for subset_type in ['train', 'val', 'test']:
    print(subset_type)
    # path to pdf to generate
    filename = os.path.join(MIP_folder, subset_type, 'MIP_preprocessed_{}_data_threshold_{}.pdf'.format(subset_type, str(threshold)))
    print('folder :', os.path.join(MIP_folder, subset_type))
    print('filename :', filename)
    if not os.path.exists(os.path.join(MIP_folder, subset_type)):
        os.makedirs(os.path.join(MIP_folder, subset_type))
        print(os.path.join(MIP_folder, subset_type), 'folder created')

    # Define generator
    generator = DataGenerator(dataset[subset_type]['x'], dataset[subset_type]['y'],
                              batch_size=batch_size, shuffle=False, augmentation=False,
                              target_shape=image_shape, target_voxel_spacing=voxel_spacing,
                              resize=resize, normalize=normalize, threshold=threshold)

    with PdfPages(filename) as pdf:

        # loop on files to get MIP visualisation
        for step, (X, Mask) in enumerate(generator):
            print(step)
            _, suptitle = ntpath.split(dataset[subset_type]['x'][step][0])  # PET file name
            PET_scan = X[0, :, :, :, 0]
            CT_scan = X[0, :, :, :, 1]
            Mask = Mask[0]

            # for TEP visualisation
            PET_scan = np.where(PET_scan > 1.0, 1.0, PET_scan)
            PET_scan = np.where(PET_scan < 0.0, 0.0, PET_scan)

            # for CT visualisation
            CT_scan = np.where(CT_scan > 1.0, 1.0, CT_scan)
            CT_scan = np.where(CT_scan < 0.0, 0.0, CT_scan)

            # # for correct visualisation
            # PET_scan = np.flip(PET_scan, axis=0)
            # CT_scan = np.flip(CT_scan, axis=0)
            # Mask = np.flip(Mask, axis=0)

            # stacked projections
            PET_scan = np.hstack((np.amax(PET_scan, axis=1), np.amax(PET_scan, axis=2)))
            CT_scan = np.hstack((np.amax(CT_scan, axis=1), np.amax(CT_scan, axis=2)))
            Mask = np.hstack((np.amax(Mask, axis=1), np.amax(Mask, axis=2)))

            # Plot
            f = plt.figure(figsize=(15, 10))
            f.suptitle(suptitle, fontsize=15)
            # f.suptitle('splitext(basename(PET_id))[0)', fontsize=15)

            plt.subplot(121)
            plt.imshow(CT_scan, cmap=color_CT, origin='lower')
            plt.imshow(PET_scan, cmap=color_PET, alpha=transparency, origin='lower')
            plt.axis('off')
            plt.title('PET/CT', fontsize=20)

            plt.subplot(122)
            plt.imshow(CT_scan, cmap=color_CT, origin='lower')
            plt.imshow(PET_scan, cmap=color_PET, alpha=transparency, origin='lower')
            plt.imshow(np.where(Mask, 0, np.nan), cmap=color_MASK, origin='lower')
            plt.axis('off')
            plt.title('PET/CT + Segmentation', fontsize=20)

            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
