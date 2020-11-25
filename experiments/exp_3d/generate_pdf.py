import argparse
from lib.utils import read_cfg

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .preprocessing import get_data
from lib.visualize import display_instance
from tqdm import tqdm


def main(cfg, args):

    dataset, train_transforms, val_transforms = get_data(cfg)

    for subset, data in dataset.items():
        filename = args.filename + '_' + subset + '.pdf'
        with PdfPages(filename) as pdf:
            for count, img_path in enumerate(tqdm(data)):
                result_dict = val_transforms(img_path)
                study_uid = result_dict['image_id']
                spacing = result_dict['meta_info']['new_spacing']

                pet_array = result_dict['input'][:, :, :, 0]
                ct_array = result_dict['input'][:, :, :, 1]
                mask_array = np.round(result_dict['mask_img'])

                pet_array = np.flip(pet_array, axis=0)
                ct_array = np.flip(ct_array, axis=0)
                mask_array = np.flip(mask_array, axis=0)

                PET_scan = np.hstack((np.max(pet_array, axis=1), np.max(pet_array, axis=2)))
                CT_scan = np.hstack((np.max(ct_array, axis=1), np.max(ct_array, axis=2)))

                figsize = (16, 16)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
                # Title for whole figure
                tmtv = np.sum(mask_array) * np.prod(spacing) * 10**-6
                fig.suptitle("id : {}\n tmtv {} mL".format(study_uid, tmtv), fontsize=16)

                # Left fig plot
                ax1.imshow(CT_scan, cmap=plt.cm.gray, origin='lower')
                ax1.imshow(PET_scan, cmap=plt.cm.plasma, alpha=0.7, origin='lower')
                ax1.set_title('PET/CT scan')

                # right fig polot
                display_instance(pet_array, mask_array=mask_array, axis=1, ax=ax2)
                ax1.set_title('PET & seg')

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
