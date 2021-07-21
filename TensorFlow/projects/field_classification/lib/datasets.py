import numpy as np 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataManager:
    """A class to manage data and prepare train, val and test dataset in list of dict
    """

    def __init__(self, csv_path):
        """prepare DataManager object

        Args:
            csv_path ([str]): [columns : 'STUDY_UID', 'NIFTI_CT', 'UPPER_LIMIT', 'LOWER_LIMIT', 'RIGHT_ARM', 'LEFT_ARM']
        """
        self.csv_path = csv_path
        self.seed = 42
        self.test_size = 0.15 
        self.val_size = 0.15 

    def get_train_val_test(self):
        df = pd.read_csv(self.csv_path)

        key_split = 'STUDY_UID'  # unique id
        idx = np.arange(df[key_split].nunique()) #0 to number of PET0
        split = np.empty(df[key_split].nunique(), dtype="<U6")

        idx_train, idx_test = train_test_split(idx, test_size=self.test_size, random_state=self.seed) #index 
        idx_train, idx_val = train_test_split(idx_train, test_size=self.val_size, random_state=self.seed) #index 

        split[idx_train] = 'train'
        split[idx_val] = 'val'
        split[idx_test] = 'test' #put at avery index train, test, val
                #split = array avec train val test aux index correspondant

        df_patient = pd.DataFrame(data={key_split: df[key_split].unique(),
                                                'subset': split})
        df = df.merge(df_patient, on=key_split, how='left')
                #add column subset on the DataFrame/CSV

        df_train = df[df['subset'] == 'train']
        df_val = df[df['subset'] == 'val']
        df_test = df[df['subset'] == 'test']
        return self.wrap_in_list_of_dict(df_train), self.wrap_in_list_of_dict(df_val), self.wrap_in_list_of_dict(df_test)

    @staticmethod
    def wrap_in_list_of_dict(df):
        """
        :return: [ {'ct_img': ct_img0_path, 'upper_limit': value, 'lower_limit':value ,'right_arm' : value, 'left_arm':value, }
                    {'ct_img': ct_img1_path, 'upper_limit': value, 'lower_limit':value ,'right_arm' : value, 'left_arm':value, }
                    {'ct_img': ct_img2_path, 'upper_limit': value, 'lower_limit':value ,'right_arm' : value, 'left_arm':value, }  ...]
        """
        mapper = {'STUDY_UID':'study_id', 'NIFTI_CT': 'ct_img', 'UPPER_LIMIT':'upper_limit', 'LOWER_LIMIT':'lower_limit', 'RIGHT_ARM': 'right_arm', 'LEFT_ARM':'left_arm'}
        return df[['STUDY_UID', 'NIFTI_CT', 'UPPER_LIMIT', 'LOWER_LIMIT', 'RIGHT_ARM', 'LEFT_ARM']].rename(columns=mapper).to_dict('records')


def get_data(csv_path):
    """prepare train,val and test dataset from a csv

    Args:
        csv_path ((str)): [path of the csv file]

    Returns:
        [dict]: [return train, val and test dataset with dict]
    """

    DM = DataManager(csv_path)
    train, val, test = DM.get_train_val_test()
    dataset = dict()
    dataset['train'], dataset['val'], dataset['test'] = train, val, test 
    return dataset 




