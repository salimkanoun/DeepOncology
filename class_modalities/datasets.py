import numpy as np
import os
import glob

from sklearn.model_selection import train_test_split


"""
        PET_ids = np.sort(glob.glob(directory + '/*float*.nii'))
        CT_ids = np.sort(glob.glob(directory + '/*ctUh*.nii'))
        MASK_ids = np.sort(glob.glob(directory + '/*pmask*.nii'))

        preprocessed_sets[folder] = list(zip(PET_ids, CT_ids, MASK_ids))

"""

class DataManager(object):

    def __init__(self, base_path):
        self.base_path = base_path
        self.seed = 42

    def get_data(self):

        PET_ids = np.sort(glob.glob(os.path.join(self.base_path, '*_nifti_PT.nii')))
        CT_ids = np.sort(glob.glob(os.path.join(self.base_path, '*_nifti_CT.nii')))
        MASK_ids = np.sort(glob.glob(os.path.join(self.base_path, '*_nifti_mask.nii')))
        return list(zip(PET_ids, CT_ids)), MASK_ids

    def get_train_val_test(self):
        X, y = self.get_data()
        return self.split_train_val_test_split(X, y, random_state=self.seed)


    @staticmethod
    def split_train_val_test_split(X, y, test_size=0.15, val_size=0.15, random_state=42):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        size = val_size/(1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=size, random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test



