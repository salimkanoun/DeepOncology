import argparse

import csv
from tqdm import tqdm

import numpy as np
import pandas as pd
import SimpleITK as sitk
# import tensorflow as tf
from experiments.exp_3d.preprocessing import *
from losses.Metrics import metric_dice

from lib.utils import read_cfg
from lib.visualize import display_instance
from .inference import Pipeline

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main(cfg, args):
    df = pd.read_csv(args.files)
    dataset = df[['study_uid', 'mask_img', 'ct_img']].to_dict('records')

    if args.weight == '':
        last = sorted(os.listdir('/media/oncopole/DD 2To/RUDY_WEIGTH/training'))[-1]
        model_path = os.path.join(cfg['training_model_folder'], last,
                                  'model_weights.h5')
    else:
        model_path = args.weight

    pipeline = Pipeline(cfg, model_path=model_path)


    filename = args.filename
    with PdfPages(filename) as pdf:
        for count, img_path in enumerate(dataset):
            study_uid = img_path['study_id']
            pet_img = sitk.ReadImage(img_path['pet_img'], sitk.sitkFloat32)
            pet_array = sitk.GetArrayFromImage(pet_img)

            pred_nifti = pipeline(img_path)
            pred = sitk.GetArrayFromImage(pred_nifti)

            # coronal
            axis = 1
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))
            fig.suptitle("id : {}".format(study_uid))

            display_instance(pet_array, mask_array=None, axis=axis, ax=ax1)
            ax1.set_title('PET scan')

            display_instance(pet_array, mask_array=pred, axis=axis, ax=ax2)
            ax2.set_title('prediction')

            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            # sagital
            axis = 2
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))
            fig.suptitle("id : {}".format(study_uid))

            display_instance(pet_array, mask_array=None, axis=axis, ax=ax1)
            ax1.set_title('PET scan')

            display_instance(pet_array, mask_array=pred, axis=axis, ax=ax2)
            ax2.set_title('prediction')

            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()











if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/config_3d.py', type=str,
                        help="path/to/config.py")
    parser.add_argument('-w', "--weight", default='', type=str,
                        help='path/to/model/weight.h5')
    parser.add_argument("-t", "--target_filename", type=str,
                        help="path/to/target/file.pdf where to save the pdf file")
    parser.add_argument("-f", "--files", type=str,
                        help="path/to/files.csv")
    args = parser.parse_args()

    config = read_cfg(args.config)
    config['cfg_path'] = args.config



    main(config, args)
