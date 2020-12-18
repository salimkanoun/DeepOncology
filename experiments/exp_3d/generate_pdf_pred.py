import argparse

import csv
from tqdm import tqdm

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf
from experiments.exp_3d.preprocessing import *
from losses.Metrics import metric_dice

from lib.utils import read_cfg
from lib.visualize import display_instance, plot_seg, plot_diff
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

    dataset, train_transforms, val_transforms = get_data(cfg)
    model = tf.keras.models.load_model(model_path, compile=False)

    filename = args.filename
    with PdfPages(filename) as pdf:
        for count, img_path in enumerate(dataset):
            study_uid = img_path['study_id']
            img_dict = val_transforms(img_path)

            img = img_dict['input']
            pet_array = np.squeeze(img)[:, :, :, 0]
            img = np.expand_dims(img, axis=0)

            gt = img_dict['mask_img']
            gt = np.squeeze(np.round(gt))

            pred = model.predict(img)
            pred = np.squeeze(np.round(pred))

            # coronal
            plot_seg(pet_array, gt, pred, axis=1, suptitle=study_uid)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            # sagital
            plot_seg(pet_array, gt, pred, axis=2, suptitle=study_uid)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            # difference
            plot_diff(pet_array, gt, pred, axis=1, ax=None)
            plt.close()


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
