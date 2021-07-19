
from os import PRIO_PGRP
from dicom_to_cnn.model.reader.Nifti import Nifti 
from dicom_to_cnn.model.post_processing.mip.MIP_Generator import MIP_Generator 
from torch import nn 
import torch 
import numpy as np
from .lib.encoding_instance import encoding_instance

class Transformed_dataset():
    """Pytorch dataset."""
    def __init__(self, list_data, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.list_data = list_data
        self.transform = transform
        self.i = 0

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):
        self.i+=1
        #print("Chargement image", self.i, " ...")
        if torch.is_tensor(idx):
            idx = idx.tolist()
        labels = encoding_instance(self.list_data[idx])
        sample = {'image': self.list_data[idx][1], 'head': labels[0], 'leg' : labels[1], 'right_arm': labels[2], 'left_arm': labels[3]}
        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale_transform_array(object):
    def __init__(self, output_shape=(256,256,1024), angle = 0):
        self.output_shape = output_shape
        self.angle = angle

    def __call__(self, sample):
        objet = Nifti(sample['image'])
        resampled = objet.resample(shape=self.output_shape)
        mip_generator = MIP_Generator(resampled)
        array=mip_generator.project(angle=self.angle)
        array[np.where(array < 500)] = 0 #500 UH
        array[np.where(array > 1024)] = 1024 #1024 UH
        array = array[:,:,]/1024
        array = np.expand_dims(array, axis=0)
        sample  = {'image': torch.from_numpy(np.array(array)), 'head': torch.from_numpy(np.array(sample['head'])),'leg': torch.from_numpy(np.array(sample['leg'])),'right_arm': torch.from_numpy(np.array(sample['right_arm'])),'left_arm': torch.from_numpy(np.array(sample['left_arm'])) }
        return sample
