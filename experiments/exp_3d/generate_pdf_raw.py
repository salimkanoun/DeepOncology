import argparse
from lib.utils import read_cfg

import numpy as np
import SimpleITK as sitk

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

from .preprocessing import get_data
from lib.visualize import display_instance
from tqdm import tqdm

from lib.datasets import DataManager
from lib.transforms import *
from lib.visualize import display_roi


def get_data(cfg):
    csv_path = cfg['csv_path']

    # Get Data
    DM = DataManager(csv_path=csv_path)
    train_images_paths, val_images_paths, test_images_paths = DM.get_train_val_test(wrap_with_dict=True)
    dataset = dict()
    dataset['train'], dataset['val'], dataset['test'] = train_images_paths, val_images_paths, test_images_paths

    # Define generator
    keys = ('pet_img', 'ct_img', 'mask_img')
    transforms = LoadNifti(keys=keys)

    return dataset, transforms


def main(cfg, args):

    dataset, loader = get_data(cfg)

    for subset, data in dataset.items():
        filename = args.filename + '_' + subset + '.pdf'
        with PdfPages(filename) as pdf:
            for count, img_path in enumerate(tqdm(data)):
                result_dict = loader(img_path)
                study_uid = result_dict['image_id']

                pet_img = result_dict['pet_img']
                ct_img = result_dict['ct_img']
                mask_img = result_dict['mask_img']

                pet_array = sitk.GetArrayFromImage(pet_img)
                ct_array = sitk.GetArrayFromImage(ct_img)
                mask_array = sitk.GetArrayFromImage(mask_img)

                pet_array = np.clip(pet_array, 0.0, 10.0)
                ct_array = np.clip(ct_array, -1000.0, 1000.0)

                # pet_array = np.flip(pet_array, axis=0)
                # ct_array = np.flip(ct_array, axis=0)
                # mask_array = np.flip(mask_array, axis=1)

                PET_scan = np.hstack((np.max(pet_array, axis=1),
                                      np.max(pet_array, axis=2)))
                # CT_scan = np.hstack(np.max(ct_array[:, int(ct_array.shape[1]//2), :], axis=1),
                #                     np.max(ct_array[:, :, int(ct_array.shape[2]//2)], axis=2))
                CT_scan = np.hstack(ct_array[:, int(ct_array.shape[1]//2), :],
                                    ct_array[:, :, int(ct_array.shape[2]//2)])

                figsize = (16, 16)
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
                # Title for whole figure
                fig.suptitle("id : {}".format(study_uid), fontsize=16)

                # PET plot
                im1 = ax1.imshow(PET_scan)
                # add color bar
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im1, cax=cax, orientation='vertical')
                ax1.set_title('PET scan\n spacing : {}, size {}'.format(pet_img.GetSpacing(), pet_img.GetSize()))
                # CT plot
                im2 = ax2.imshow(CT_scan)
                # add color bar
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im2, cax=cax, orientation='vertical')
                ax2.set_title('CT scan\n spacing : {}, size {}'.format(ct_img.GetSpacing(), ct_img.GetSize()))

                # ROI plot
                display_roi(pet_array, mask_array, axis=1, ax=ax3)
                ax3.set_title('VOI scan\n n_VOI= {}'.format(mask_array.shape[0]))

                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/config_3d.py', type=str,
                        help="path/to/config.py")
    parser.add_argument("-f", "--filename", default='dummy', type=str,
                        help="path/to/where/to/save/filename")
    # parser.add_argument("-s", "--subset", default='all', type=str,
    #                     help="path/to/where/to/save/filename")
    args = parser.parse_args()

    config = read_cfg(args.config)

    main(config, args)
