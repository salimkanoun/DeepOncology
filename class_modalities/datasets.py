import numpy as np
import pandas as pd
import os
import glob

from sklearn.model_selection import train_test_split


class DataManager(object):

    def __init__(self, base_path=None, csv_path=None):
        self.base_path = base_path
        self.csv_path = csv_path
        self.seed = 42  # random state
        self.test_size = 0.15
        self.val_size = 0.15

    def get_data(self):

        PET_ids = np.sort(glob.glob(os.path.join(self.base_path, '*_nifti_PT.nii')))
        CT_ids = np.sort(glob.glob(os.path.join(self.base_path, '*_nifti_CT.nii')))
        MASK_ids = np.sort(glob.glob(os.path.join(self.base_path, '*_nifti_mask.nii')))
        return list(zip(PET_ids, CT_ids)), MASK_ids

    def get_train_val_test(self):
        if self.csv_path is None:
            X, y = self.get_data()
            return self.split_train_val_test_split(X, y, random_state=self.seed)
        else:
            df = pd.read_csv(self.csv_path)
            if 'split' not in df.columns:
                idx = np.arange(df['PATIENT NAME'].nunique())
                split = np.empty(df['PATIENT NAME'].nunique(), dtype="<U6")

                idx_train, idx_test = train_test_split(idx, test_size=self.test_size, random_state=self.seed)

                size = self.val_size / (1 - self.test_size)
                idx_train, idx_val = train_test_split(idx_train, test_size=size, random_state=self.seed)

                split[idx_train] = 'train'
                split[idx_val] = 'val'
                split[idx_test] = 'test'

                df_patient = pd.DataFrame(data={'PATIENT NAME': df['PATIENT NAME'].unique(),
                                                'split': split})
                df = df.merge(df_patient, on='PATIENT NAME', how='left')

            df_train = df[df['split'] == 'train']
            df_val = df[df['split'] == 'val']
            df_test = df[df['split'] == 'test']

            X_train, y_train = list(zip(df_train['NIFTI_PET'].values, df_train['NIFTI_CT'].values)), df_train['NIFTI_MASK'].values
            X_val, y_val = list(zip(df_val['NIFTI_PET'].values, df_val['NIFTI_CT'].values)), df_val['NIFTI_MASK'].values
            X_test, y_test = list(zip(df_test['NIFTI_PET'].values, df_test['NIFTI_CT'].values)), df_test['NIFTI_MASK'].values

            return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def wrap_in_dict(df):
        # df.T.to_dict().values()
        return {'pet_img': df['NIFTI_PET'].values,
                'ct_img': df['NIFTI_CT'].values,
                'mask_img': df['NIFTI_MASK'].values}

    @staticmethod
    def wrap_in_list_of_dict(df):
        # retur df[['NIFTI_PET', 'NIFTI_CT', 'NIFTI_MASK']].T.to_dict().values()
        return df[['NIFTI_PET', 'NIFTI_CT', 'NIFTI_MASK']].to_dict('records')

    @staticmethod
    def split_train_val_test_split(X, y, test_size=0.15, val_size=0.15, random_state=42):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        size = val_size/(1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=size, random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test



