import os
from glob import glob

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

    def get_data(self):
        pass

    def get_train_val_test(self):
        pass

    @staticmethod
    def split_train_val_test_split(X, y, test_size=0.15, val_size=0.15):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        size = val_size/(1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=size, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test



